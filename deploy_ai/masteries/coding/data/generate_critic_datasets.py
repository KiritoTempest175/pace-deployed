import os
import pandas as pd
import numpy as np

# 1. Define paths
RAW_PARQUET_PATH = "masteries/coding/data/raw/stack_python_157k.parquet"
OUTPUT_DIR = "masteries/coding/data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading the master dataset...")
df_master = pd.read_parquet(RAW_PARQUET_PATH)

# 2. SHUFFLE FIRST: Destroy any original order in the dataset
print("Performing initial shuffle...")
df_master = df_master.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. THE ROUND-ROBIN ALLOCATOR
# We use modulo 6 to perfectly zipper the data (50% Clean, 50% Bugs spread across 3 types)
# Pattern: 0=Clean, 1=Flip, 2=Clean, 3=Constant, 4=Clean, 5=Deletion
print("Slicing into mutually exclusive categories...")
conditions = [
    (df_master.index % 6 == 0) | (df_master.index % 6 == 2) | (df_master.index % 6 == 4), # 50% Clean
    (df_master.index % 6 == 1), # 16.6% Flip
    (df_master.index % 6 == 3), # 16.6% Constant
    (df_master.index % 6 == 5)  # 16.6% Deletion
]
choices = ['CLEAN', 'FLIP', 'CONSTANT', 'DELETION']
df_master['assigned_type'] = np.select(conditions, choices, default='UNKNOWN')

# Isolate the slices
df_clean = df_master[df_master['assigned_type'] == 'CLEAN'].copy()
df_flips = df_master[df_master['assigned_type'] == 'FLIP'].copy()
df_constants = df_master[df_master['assigned_type'] == 'CONSTANT'].copy()
df_deletions = df_master[df_master['assigned_type'] == 'DELETION'].copy()

# ==========================================
# APPLY MUTATIONS (Safely isolated)
# ==========================================
print(f"Applying mutations (Zero Overlap Guaranteed)...")

# FLIPS
df_flips["mutated_code"] = (
    df_flips["content"]
    .str.replace(" == ", " != ", regex=False)
    .str.replace(" < ", " > ", regex=False)
    .str.replace(" + ", " - ", regex=False)
)
df_flips["bug_type"] = "OPERATOR_FLIP"
df_flips["label"] = "BUG"
df_flips = df_flips[df_flips["mutated_code"] != df_flips["content"]].copy()

# CONSTANTS
df_constants["mutated_code"] = (
    df_constants["content"]
    .str.replace(" 0", " 1", regex=False)
    .str.replace("True", "False", regex=False)
    .str.replace("False", "True", regex=False)
)
df_constants["bug_type"] = "CONSTANT_SHIFT"
df_constants["label"] = "BUG"
df_constants = df_constants[df_constants["mutated_code"] != df_constants["content"]].copy()

# DELETIONS
def drop_return_statement(code_str):
    lines = str(code_str).split("\n")
    surviving_lines = [line for line in lines if "return " not in line]
    return "\n".join(surviving_lines)

df_deletions["mutated_code"] = df_deletions["content"].apply(drop_return_statement)
df_deletions["bug_type"] = "STATEMENT_DELETION"
df_deletions["label"] = "BUG"
df_deletions = df_deletions[df_deletions["mutated_code"] != df_deletions["content"]].copy()

# CLEAN
df_clean["mutated_code"] = df_clean["content"]
df_clean["bug_type"] = "CLEAN"
df_clean["label"] = "CLEAN"

# ==========================================
# BALANCE THE CLASSES (50/50 MATCH)
# ==========================================
total_surviving_bugs = len(df_flips) + len(df_constants) + len(df_deletions)
print(f"Trimming clean data from {len(df_clean)} down to {total_surviving_bugs}...")
df_clean = df_clean.sample(n=total_surviving_bugs, random_state=42).copy()

# ==========================================
# RECOMBINE & FINAL SHUFFLE
# ==========================================
print("Fusing and randomizing final dataset...")
df_fused = pd.concat([df_clean, df_flips, df_constants, df_deletions], ignore_index=True)

# Remove rows sharing identical code logic (duplicates)
df_fused = df_fused.drop_duplicates(subset=["mutated_code"], keep="first").reset_index(drop=True)

# Final deep shuffle so the Critic learns no sequential patterns
df_fused = df_fused.sample(frac=1, random_state=99).reset_index(drop=True)

# Save
fused_path = os.path.join(OUTPUT_DIR, "critic_fused_dataset.parquet")
df_fused.to_parquet(fused_path)

print(f"SUCCESS! Dataset saved to {fused_path}")
print(f"Total Rows: {len(df_fused)}")
print(f"Clean Data: {(df_fused['label'] == 'CLEAN').sum()} | Bug Data: {(df_fused['label'] == 'BUG').sum()}")