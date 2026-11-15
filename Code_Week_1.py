from datasets import load_dataset

# Download and load the IMDb dataset
dataset = load_dataset("stanfordnlp/imdb")


# First five reviews
for i in range(5):
    print("Review:", dataset["train"][i]["text"])
    print("Sentiment:", dataset["train"][i]["label"])
    print("-" * 80)

pippeli :3