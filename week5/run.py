"""CLI orchestration. Individual completed experiments can be reused safely."""
import argparse
import json
from pathlib import Path

# Support both the repository entry point and running this file directly.
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "week5"

from .common import ROOT, prepare_data, set_seed, save_json
import torch
from filelock import FileLock
from threadpoolctl import threadpool_limits


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["all", "prepare", "imdb", "banking77", "transfer", "report"],
                        default="all")
    parser.add_argument("--quick", action="store_true", help="Small real-data smoke run in a separate folder")
    parser.add_argument("--dataset", choices=["imdb", "banking77", "both"], default="banking77",
                        help="Dataset for the transfer-learning layer study")
    parser.add_argument("--checkpoint", default="prajjwal1/bert-tiny",
                        help="Pretrained BERT checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scratch-lr", type=float, default=5e-4)
    parser.add_argument("--transfer-lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, help="Fresh output directory for changed configurations")
    return parser.parse_args()


def main():
    args = arguments()
    for name in ("epochs", "batch_size", "max_length", "threads", "patience"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if args.max_length < 4:
        raise ValueError("max-length must leave room for special tokens.")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu.")
    config = {
        "seed": args.seed, "epochs": 1 if args.quick else args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "patience": args.patience, "scratch_lr": args.scratch_lr,
        "transfer_lr": args.transfer_lr, "checkpoint": args.checkpoint,
        "device": device, "quick": args.quick, "threads": args.threads,
        "transfer_dataset": args.dataset,
    }
    output = args.output or ROOT / "output" / "week5" / ("smoke" if args.quick else "full")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=0), threadpool_limits(limits=args.threads):
        config_path = output / "config.json"
        if config_path.exists():
            recorded = json.loads(config_path.read_text())
            if args.part == "report":
                config = recorded
            elif config != recorded:
                raise ValueError("Output contains a different configuration. Choose a fresh --output folder.")
        save_json(config_path, config)
        print(f"Device: {device}; output: {output}", flush=True)
        if args.part == "report":
            from .report import generate_report
            generate_report(output, config)
            return

        transfer_names = ("imdb", "banking77") if args.dataset == "both" else (args.dataset,)
        names = ("imdb", "banking77") if args.part in ("all", "prepare") else (
            transfer_names if args.part == "transfer" else (args.part,)
        )
        from .visualize import dataset_plots
        from .baseline import run_baseline
        from .transformer import TransformerClassifier, encode_frames
        from .training import train_model
        from .transfer import MODES, encode_pretrained, create_model, checkpoint_revision
        for name in names:
            print(f"Preparing {name}...", flush=True)
            frames, metadata = prepare_data(name, args.seed, args.quick)
            metadata_path = output / name / "dataset.json"
            if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
                raise ValueError("Dataset fingerprints or splits changed. Choose a fresh --output folder.")
            save_json(metadata_path, metadata)
            dataset_plots(frames, metadata, output / name)
            if args.part == "prepare":
                continue
            if args.part != "transfer":
                folder = output / name / "baseline"
                if not (folder / "result.json").exists():
                    print(f"Training {name} baseline...", flush=True)
                    run_baseline(frames, metadata, folder, config)
                folder = output / name / "scratch"
                if not (folder / "result.json").exists():
                    set_seed(config["seed"])
                    vocabulary, datasets, truncated = encode_frames(frames, config["max_length"])
                    model = TransformerClassifier(len(vocabulary), metadata["classes"], config["max_length"])
                    train_model(model, datasets, frames, metadata, folder, config, "scratch",
                                {"vocabulary": vocabulary, "truncated_fraction": truncated})
            if args.part in ("all", "transfer") and name in transfer_names:
                tokenizer, datasets, truncated = encode_pretrained(frames, args.checkpoint, config["max_length"])
                for mode in MODES:
                    folder = output / name / f"transfer_{mode}"
                    if (folder / "result.json").exists():
                        continue
                    # Reset before constructing each classifier and its data-order generator.
                    set_seed(config["seed"])
                    model = create_model(args.checkpoint, metadata["label_names"], mode)
                    if config["max_length"] > model.config.max_position_embeddings:
                        raise ValueError("max-length exceeds the pretrained checkpoint's position limit.")
                    folder.mkdir(parents=True, exist_ok=True)
                    tokenizer.save_pretrained(folder / "tokenizer")
                    model.config.save_pretrained(folder)
                    train_model(model, datasets, frames, metadata, folder, config, f"transfer_{mode}",
                                {"checkpoint": args.checkpoint, "encoder_blocks": model.config.num_hidden_layers,
                                 "checkpoint_revision": checkpoint_revision(args.checkpoint),
                                 "truncated_fraction": truncated})
        if args.part == "prepare":
            # Cache checkpoint/tokenizer in advance so training can run offline.
            from transformers import BertTokenizer, BertModel
            BertTokenizer.from_pretrained(args.checkpoint)
            BertModel.from_pretrained(args.checkpoint)
        from .report import generate_report
        generate_report(output, config)


if __name__ == "__main__":
    main()
