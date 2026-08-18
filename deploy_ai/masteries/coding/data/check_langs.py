import datasets

def main():
    ds = datasets.load_dataset("ByteDance-Seed/Code-Contests-Plus", split="train", streaming=True)
    langs = set()
    for i, item in enumerate(ds):
        for sub in item.get('correct_submissions', []):
            langs.add(sub.get('language', ''))
        for sub in item.get('incorrect_submissions', []):
            langs.add(sub.get('language', ''))
        if i >= 100:
            break
    print("Available languages:", langs)

if __name__ == "__main__":
    main()
