import pandas as pd

df = pd.read_parquet("masteries/coding/data/raw/stack_python_157k.parquet")


print("\n--- Here is Row 0: The First Code Example! ---")
print("FUNCTION CODE:")
print(df["content"].iloc[0])  # iloc[0] means "Index Location 0" (the first row)

print("\nAUTOMATED UNIT TESTS:")
print(df["tests"].iloc[0])