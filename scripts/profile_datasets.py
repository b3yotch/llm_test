from datasets import load_dataset
from collections import Counter
import json
import os
import re


TRENDYOL = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
SHAREGPT = "ChaoticNeutrals/Cybersecurity-ShareGPT"

REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower()
    ).strip()


def text_length(value):
    return len(str(value))


def get_text_stats(values):
    lengths = [
        text_length(value)
        for value in values
        if value is not None
    ]

    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "avg": 0,
            "median": 0,
        }

    sorted_lengths = sorted(lengths)

    middle = len(sorted_lengths) // 2

    if len(sorted_lengths) % 2:
        median = sorted_lengths[middle]
    else:
        median = (
            sorted_lengths[middle - 1]
            + sorted_lengths[middle]
        ) / 2

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths),
        "median": median,
    }


def print_stats(field, values):
    stats = get_text_stats(values)

    empty = sum(
        1
        for value in values
        if not normalize_text(value)
    )

    print(f"\n{field}")
    print(f"  min chars    : {stats['min']}")
    print(f"  max chars    : {stats['max']}")
    print(f"  avg chars    : {stats['avg']:.2f}")
    print(f"  median chars : {stats['median']}")
    print(f"  empty        : {empty}")


def exact_duplicate_count(dataset, fields):
    seen = set()
    duplicates = 0

    for row in dataset:
        key = tuple(
            normalize_text(row[field])
            for field in fields
        )

        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    return duplicates


def contains_code(text):
    """
    Heuristic only.

    This does NOT determine whether an example is good or bad.
    It simply identifies examples that look like they contain code.
    """

    text = str(text)

    patterns = [
        r"```",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\s+",
        r"\bSELECT\s+.+\bFROM\b",
        r"\bfunction\s+\w+\s*\(",
        r"\bconst\s+\w+\s*=",
        r"\bvar\s+\w+\s*=",
        r"\bpublic\s+class\s+\w+",
        r"\b#include\s*<",
        r"\$\w+\s*=",
        r"#!/usr/bin/",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        for pattern in patterns
    )


def contains_generic_programming_signals(text):
    """
    Heuristic only.

    Used to identify examples that MAY be generic programming
    rather than cybersecurity-focused.

    These examples are NOT automatically removed.
    """

    text = normalize_text(text)

    signals = [
        "write a python program",
        "write a python script",
        "python tutorial",
        "learn python",
        "python programming",
        "javascript tutorial",
        "write a javascript program",
        "write a java program",
        "write a c++ program",
        "implement a calculator",
        "build a web scraper",
        "create a web scraper",
        "sort a list",
        "reverse a string",
        "fibonacci",
        "binary search",
        "linked list",
    ]

    return any(
        signal in text
        for signal in signals
    )


# ---------------------------------------------------------------------
# Trendyol profiling
# ---------------------------------------------------------------------

def profile_trendyol(dataset):

    print("\n" + "=" * 100)
    print("TRENDYOL PROFILE")
    print("=" * 100)

    print("Rows:", len(dataset))
    print("Columns:", dataset.column_names)

    report = {
        "rows": len(dataset),
        "columns": dataset.column_names,
        "fields": {},
    }

    for field in dataset.column_names:

        print_stats(
            field,
            dataset[field],
        )

        stats = get_text_stats(dataset[field])

        empty = sum(
            1
            for value in dataset[field]
            if not normalize_text(value)
        )

        report["fields"][field] = {
            **stats,
            "empty": empty,
        }

    # -------------------------------------------------------------
    # Exact duplicates
    # -------------------------------------------------------------

    duplicates = exact_duplicate_count(
        dataset,
        ["user", "assistant"],
    )

    print("\nExact duplicate user/assistant pairs:", duplicates)

    report["exact_duplicate_user_assistant_pairs"] = duplicates

    # -------------------------------------------------------------
    # Duplicate users
    # -------------------------------------------------------------

    users = [
        normalize_text(value)
        for value in dataset["user"]
    ]

    duplicate_users = len(users) - len(set(users))

    print("Repeated user questions:", duplicate_users)

    report["repeated_user_questions"] = duplicate_users

    return report


# ---------------------------------------------------------------------
# ShareGPT profiling
# ---------------------------------------------------------------------

def find_sharegpt_text_fields(dataset):

    """
    Try to determine likely conversation/text fields without
    assuming a particular schema.
    """

    candidates = [
        "prompt",
        "instruction",
        "input",
        "output",
        "user",
        "assistant",
        "conversation",
        "conversations",
        "messages",
        "text",
    ]

    return [
        field
        for field in candidates
        if field in dataset.column_names
    ]


def profile_sharegpt(dataset):

    print("\n" + "=" * 100)
    print("CYBERSECURITY-SHAREGPT PROFILE")
    print("=" * 100)

    print("Rows:", len(dataset))
    print("Columns:", dataset.column_names)

    report = {
        "rows": len(dataset),
        "columns": dataset.column_names,
        "fields": {},
    }

    # -------------------------------------------------------------
    # Basic field statistics
    # -------------------------------------------------------------

    for field in dataset.column_names:

        print_stats(
            field,
            dataset[field],
        )

        stats = get_text_stats(dataset[field])

        empty = sum(
            1
            for value in dataset[field]
            if not normalize_text(value)
        )

        report["fields"][field] = {
            **stats,
            "empty": empty,
        }

    # -------------------------------------------------------------
    # Inspect likely text fields
    # -------------------------------------------------------------

    text_fields = find_sharegpt_text_fields(dataset)

    print("\nLikely conversation/text fields:")
    for field in text_fields:
        print("  -", field)

    report["likely_text_fields"] = text_fields

    # -------------------------------------------------------------
    # Code detection
    # -------------------------------------------------------------

    code_count = 0

    for row in dataset:

        combined = " ".join(
            str(row[field])
            for field in text_fields
            if field in row
        )

        if contains_code(combined):
            code_count += 1

    print("\nExamples containing likely code:", code_count)

    report["likely_code_examples"] = code_count

    # -------------------------------------------------------------
    # Generic programming detection
    # -------------------------------------------------------------

    generic_count = 0

    for row in dataset:

        combined = " ".join(
            str(row[field])
            for field in text_fields
            if field in row
        )

        if contains_generic_programming_signals(combined):
            generic_count += 1

    print(
        "Examples with generic-programming signals:",
        generic_count,
    )

    report["possible_generic_programming"] = generic_count

    # -------------------------------------------------------------
    # Exact duplicate detection
    # -------------------------------------------------------------

    if "user" in dataset.column_names and "assistant" in dataset.column_names:

        duplicates = exact_duplicate_count(
            dataset,
            ["user", "assistant"],
        )

    elif "prompt" in dataset.column_names and "response" in dataset.column_names:

        duplicates = exact_duplicate_count(
            dataset,
            ["prompt", "response"],
        )

    else:

        # Fallback: hash all fields
        seen = set()
        duplicates = 0

        for row in dataset:

            key = tuple(
                normalize_text(row[field])
                for field in dataset.column_names
            )

            if key in seen:
                duplicates += 1
            else:
                seen.add(key)

    print("\nExact duplicate examples:", duplicates)

    report["exact_duplicates"] = duplicates

    return report


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

print("=" * 100)
print("LOADING DATASETS")
print("=" * 100)

trendyol = load_dataset(
    TRENDYOL,
    split="train",
)

sharegpt = load_dataset(
    SHAREGPT,
    split="train",
)


trendyol_report = profile_trendyol(trendyol)

sharegpt_report = profile_sharegpt(sharegpt)


# ---------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------

report = {
    "trendyol": trendyol_report,
    "sharegpt": sharegpt_report,
}


report_path = os.path.join(
    REPORT_DIR,
    "dataset_profile.json",
)

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 100)
print("PROFILE COMPLETE")
print("=" * 100)

print("Report saved to:")
print(report_path)