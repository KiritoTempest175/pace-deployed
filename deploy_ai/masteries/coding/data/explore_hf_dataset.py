import datasets

print("Exploring open-r1/codeforces-submissions...")
try:
    ds1 = datasets.load_dataset("open-r1/codeforces-submissions", split="train", streaming=True)
    for i, item in enumerate(ds1):
        print("Codeforces Sample:", item.keys())
        print(item)
        break
except Exception as e:
    print(f"Error loading Codeforces: {e}")

print("\nExploring ByteDance-Seed/Code-Contests-Plus...")
try:
    ds2 = datasets.load_dataset("ByteDance-Seed/Code-Contests-Plus", split="train", streaming=True)
    for i, item in enumerate(ds2):
        print("CodeContests+ Sample:", item.keys())
        # Truncate some large strings if necessary
        keys_to_print = {k: v[:200] if isinstance(v, str) else v for k, v in item.items()}
        print(keys_to_print)
        break
except Exception as e:
    print(f"Error loading CodeContests+: {e}")
