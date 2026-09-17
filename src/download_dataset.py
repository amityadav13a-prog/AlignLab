from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")

print(dataset)
print(dataset["train"][0])