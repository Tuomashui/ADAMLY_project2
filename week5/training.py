"""Shared validation-selected neural training and checkpoint saving."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .common import metrics, save_evaluation, save_json


def logits_of(model, ids, mask):
    # Trim padding within each batch without changing token positions.
    length = max(1, int(mask.sum(dim=1).max().item()))
    ids, mask = ids[:, :length], mask[:, :length]
    output = model(input_ids=ids, attention_mask=mask)
    return output.logits if hasattr(output, "logits") else output


def predict(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    total_loss = 0.0
    with torch.no_grad():
        for ids, mask, target in loader:
            logits = logits_of(model, ids.to(device), mask.to(device))
            total_loss += nn.functional.cross_entropy(logits, target.to(device), reduction="sum").item()
            probabilities.append(logits.softmax(-1).cpu().numpy())
            labels.extend(target.tolist())
    probabilities = np.concatenate(probabilities)
    return total_loss / len(labels), metrics(labels, probabilities), probabilities


def train_model(model, datasets, frames, metadata, folder, config, model_name, extra=None):
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    model.to(device)
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = {
        split: DataLoader(data, batch_size=config["batch_size"], shuffle=split == "train",
                          generator=generator if split == "train" else None)
        for split, data in datasets.items()
    }
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    save_json(folder / "trainable_parameters.json", {name: p.numel() for name, p in trainable})
    pretrained = model_name.startswith("transfer_")
    learning_rate = config["transfer_lr"] if pretrained else config["scratch_lr"]
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=learning_rate, weight_decay=0.01)
    history = []
    best_accuracy = -1.0
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        # Frozen pretrained components keep deterministic inference behavior.
        if pretrained and model_name != "transfer_full":
            model.bert.eval()
            for block in model.bert.encoder.layer:
                if any(p.requires_grad for p in block.parameters()):
                    block.train()
        running_loss = 0.0
        for ids, mask, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = logits_of(model, ids.to(device), mask.to(device))
            loss = nn.functional.cross_entropy(logits, target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            optimizer.step()
            running_loss += loss.item() * len(target)
        validation_loss, validation, _ = predict(model, loaders["validation"], device)
        history.append({
            "epoch": epoch, "train_loss": running_loss / len(datasets["train"]),
            "validation_loss": validation_loss, **validation,
        })
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(f"{metadata['dataset']} {model_name} epoch {epoch}: "
              f"validation accuracy={validation['accuracy']:.4f}", flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    probabilities = {
        split: predict(model, loaders[split], device)[2] for split in ("validation", "test")
    }
    scores = save_evaluation(folder, frames, probabilities, metadata["label_names"])
    torch.save({"state_dict": best_state, "config": config, "metadata": metadata,
                "model": model_name, "extra": extra or {}}, folder / "checkpoint.pt")
    result = {
        "dataset": metadata["dataset"], "model": model_name, "scores": scores,
        "trainable_weights": sum(p.numel() for _, p in trainable),
        "total_weights": sum(p.numel() for p in model.parameters()),
        "training_seconds": elapsed, "best_epoch": best_epoch,
        "epochs_completed": len(history), "learning_rate": learning_rate, **(extra or {}),
    }
    save_json(folder / "result.json", result)
    return result
