from datasets import load_dataset

dataset = load_dataset("nuprl/stack-dedup-python-testgen-starcoder-filter-v2")


print(dataset)
print("Saving dataset to disk... please wait a few seconds...")
dataset["train"].to_parquet("masteries/coding/data/raw/stack_python_157k.parquet")
print("SUCCESS: File permanently saved to disk!")