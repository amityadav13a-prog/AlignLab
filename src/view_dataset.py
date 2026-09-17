from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")

for i in range(10):
    item = dataset["train"][i]

    print(f"\n--- Example {i + 1} ---")
    print("Instruction:", item["instruction"])
    print("Input:", item["input"])
    print("Output:", item["output"])