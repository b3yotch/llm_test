from datasets import load_dataset

TRENDYOL = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
SHAREGPT = "ChaoticNeutrals/Cybersecurity-ShareGPT"

SAMPLE_INDICES = [0, 1, 10, 100, 1000]


def print_separator(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def inspect_dataset(name, dataset):
    print_separator(f"{name} DATASET")

    print(dataset)
    print("\nColumns:")
    for column in dataset.column_names:
        print(f"  - {column}")

    print("\nNumber of rows:", len(dataset))

    valid_indices = [
        i for i in SAMPLE_INDICES
        if i < len(dataset)
    ]

    for i in valid_indices:
        print_separator(f"{name} SAMPLE {i}")

        row = dataset[i]

        for key, value in row.items():
            print(f"\n--- {key} ---")
            print(value)


print_separator("LOADING TRENDYOL")

trendyol = load_dataset(
    TRENDYOL,
    split="train",
)

inspect_dataset("TRENDYOL", trendyol)


print_separator("LOADING CYBERSECURITY-SHAREGPT")

sharegpt = load_dataset(
    SHAREGPT,
    split="train",
)

inspect_dataset("SHAREGPT", sharegpt)


print_separator("DONE")