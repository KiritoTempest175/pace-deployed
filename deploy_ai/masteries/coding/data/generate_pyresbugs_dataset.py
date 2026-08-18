import os
import pandas as pd
from datasets import load_dataset

# 1. Define paths
OUTPUT_DIR = "masteries/coding/data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pyresbugs_fused_dataset.parquet")

print("Downloading OSS-forge/PyResBugs dataset...")
dataset = load_dataset("OSS-forge/PyResBugs")

# PyResBugs only has a train split
df_master = pd.DataFrame(dataset['train'])

print(f"Loaded {len(df_master)} raw bugs from PyResBugs.")

# 2. Extract CLEAN data (Fault Free Code)
df_clean = pd.DataFrame({
    "mutated_code": df_master["Fault Free Code"],
    "bug_type": "CLEAN",
    "label": "CLEAN"
})

# 3. Extract BUG data (Faulty Code)
df_bugs = pd.DataFrame({
    "mutated_code": df_master["Faulty Code"],
    "bug_type": df_master["Fault_Acronym"].fillna("UNKNOWN_BUG"),
    "label": "BUG"
})

# 4. Remove rows where Faulty and Fault Free code are identical (just in case)
# We can do this by merging and comparing
invalid_mask = df_master["Fault Free Code"] == df_master["Faulty Code"]
num_invalid = invalid_mask.sum()
if num_invalid > 0:
    print(f"Warning: Found {num_invalid} rows where Faulty Code == Fault Free Code. Dropping them.")
    df_clean = df_clean[~invalid_mask]
    df_bugs = df_bugs[~invalid_mask]

# Drop NaNs just in case
df_clean = df_clean.dropna(subset=["mutated_code"])
df_bugs = df_bugs.dropna(subset=["mutated_code"])

# 5. Recombine & Shuffle
print("Fusing and randomizing dataset...")
df_fused = pd.concat([df_clean, df_bugs], ignore_index=True)

# Remove completely duplicated snippets
df_fused = df_fused.drop_duplicates(subset=["mutated_code"], keep="first").reset_index(drop=True)

# Final deep shuffle
df_fused = df_fused.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_fused.to_parquet(OUTPUT_PATH)

print(f"SUCCESS! Dataset saved to {OUTPUT_PATH}")
print(f"Total Rows: {len(df_fused)}")
print(f"Clean Data: {(df_fused['label'] == 'CLEAN').sum()} | Bug Data: {(df_fused['label'] == 'BUG').sum()}")
