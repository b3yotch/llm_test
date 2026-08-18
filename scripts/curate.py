#!/usr/bin/env python3
"""
Merge cleaned ShareGPT + Trendyol cybersecurity datasets and create
a fresh train/validation split from the complete combined dataset.

Input
-----
ShareGPT (already cleaned):
    data/sharegpt_balanced/train.jsonl
    data/sharegpt_balanced/validation.jsonl

Trendyol:
    Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset

Output
------
data/final/train.jsonl
data/final/validation.jsonl
data/final/metadata.json

Pipeline
--------
1. Load ShareGPT train.
2. Load ShareGPT validation.
3. Combine both ShareGPT files into one pool.
4. Normalize ShareGPT rows.
5. Load Trendyol.
6. Normalize Trendyol rows.
7. Combine ShareGPT + Trendyol.
8. Remove exact duplicates across the complete dataset.
9. Shuffle reproducibly.
10. Create a fresh train/validation split.
11. Verify no train/validation leakage.
12. Write final datasets and metadata.

Notes
-----
- ShareGPT is assumed to already be cleaned.
- No cybersecurity topic filtering is performed here.
- No generic-programming filtering is performed here.
- No offensive-security filtering is performed here.
- Only structural validation and exact deduplication are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


# ============================================================================
# Configuration
# ============================================================================

TRENDYOL_DATASET = (
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
)

SHAREGPT_TRAIN_PATH = Path(
    "data/sharegpt_balanced/train.jsonl"
)

SHAREGPT_VALIDATION_PATH = Path(
    "data/sharegpt_balanced/validation.jsonl"
)


# ============================================================================
# Text helpers
# ============================================================================

def clean_text(value: Any) -> str:
    """
    Minimal structural normalization.

    This intentionally does not perform semantic/content filtering.
    """

    if value is None:
        return ""

    text = str(value)

    # Remove null bytes
    text = text.replace("\x00", " ")

    # Normalize Windows/Mac line endings
    text = re.sub(r"\r\n?", "\n", text)

    # Collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================================
# Row normalization
# ============================================================================

def normalize_row(
    row: dict[str, Any],
    source: str,
) -> dict[str, str] | None:
    """
    Normalize a row to:

        {
            "system": "...",
            "user": "...",
            "assistant": "...",
            "source": "..."
        }
    """

    if not isinstance(row, dict):
        return None

    system = clean_text(row.get("system", ""))
    user = clean_text(row.get("user", ""))
    assistant = clean_text(row.get("assistant", ""))

    if not user:
        return None

    if not assistant:
        return None

    return {
        "system": system,
        "user": user,
        "assistant": assistant,
        "source": source,
    }


def basic_quality_check(
    row: dict[str, str],
) -> bool:
    """
    Structural/data-quality validation only.
    """

    user = row["user"]
    assistant = row["assistant"]

    if not user.strip():
        return False

    if not assistant.strip():
        return False

    # Reject exact user == assistant examples
    if user.strip().lower() == assistant.strip().lower():
        return False

    # Reject obvious placeholder answers
    placeholder_patterns = [
        r"^\s*pass\s*$",
        r"^\s*n/?a\s*$",
        r"^\s*todo\s*$",
        r"^\s*placeholder\s*$",
    ]

    for pattern in placeholder_patterns:
        if re.fullmatch(pattern, assistant, flags=re.IGNORECASE):
            return False

    return True


# ============================================================================
# Hashing / deduplication
# ============================================================================

def normalize_for_hash(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def row_hash(row: dict[str, str]) -> str:
    """
    Hash system + user + assistant.

    Source is intentionally excluded so that an identical example appearing
    in ShareGPT and Trendyol is treated as a duplicate.
    """

    payload = (
        normalize_for_hash(row["system"])
        + "\n<SEP>\n"
        + normalize_for_hash(row["user"])
        + "\n<SEP>\n"
        + normalize_for_hash(row["assistant"])
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def deduplicate_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:

    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    duplicates = 0

    for row in rows:

        key = row_hash(row)

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        unique_rows.append(row)

    return unique_rows, duplicates


# ============================================================================
# Load local ShareGPT JSONL
# ============================================================================

def load_sharegpt_jsonl(
    path: Path,
    split_name: str,
) -> tuple[list[dict[str, str]], Counter]:

    if not path.exists():
        raise FileNotFoundError(
            f"ShareGPT file does not exist: {path}"
        )

    print()
    print("=" * 80)
    print(f"LOADING SHAREGPT {split_name.upper()}")
    print("=" * 80)
    print(f"Path: {path}")

    dataset = load_dataset(
        "json",
        data_files=str(path),
        split="train",
    )

    stats = Counter()
    rows: list[dict[str, str]] = []

    for raw_row in dataset:

        stats["input"] += 1

        row = normalize_row(
            raw_row,
            source="sharegpt_clean",
        )

        if row is None:
            stats["invalid_or_empty"] += 1
            continue

        if not basic_quality_check(row):
            stats["quality_rejected"] += 1
            continue

        rows.append(row)
        stats["kept"] += 1

    return rows, stats


# ============================================================================
# Load Trendyol
# ============================================================================

def load_trendyol() -> tuple[list[dict[str, str]], Counter]:

    print()
    print("=" * 80)
    print("LOADING TRENDYOL")
    print("=" * 80)
    print(f"Dataset: {TRENDYOL_DATASET}")

    dataset = load_dataset(
        TRENDYOL_DATASET
    )

    if isinstance(dataset, DatasetDict):

        if "train" in dataset:
            dataset = dataset["train"]
        else:
            first_split = next(iter(dataset.keys()))
            dataset = dataset[first_split]

    stats = Counter()
    rows: list[dict[str, str]] = []

    for raw_row in dataset:

        stats["input"] += 1

        row = normalize_row(
            raw_row,
            source="trendyol",
        )

        if row is None:
            stats["invalid_or_empty"] += 1
            continue

        if not basic_quality_check(row):
            stats["quality_rejected"] += 1
            continue

        rows.append(row)
        stats["kept"] += 1

    return rows, stats


# ============================================================================
# Train / validation split
# ============================================================================

def split_dataset(
    rows: list[dict[str, str]],
    validation_ratio: float,
    seed: int,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    shuffled = list(rows)

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    validation_size = max(
        1,
        int(len(shuffled) * validation_ratio),
    )

    validation = shuffled[:validation_size]
    train = shuffled[validation_size:]

    return train, validation


# ============================================================================
# Leakage verification
# ============================================================================

def verify_no_overlap(
    train: list[dict[str, str]],
    validation: list[dict[str, str]],
) -> int:

    train_hashes = {
        row_hash(row)
        for row in train
    }

    validation_hashes = {
        row_hash(row)
        for row in validation
    }

    return len(train_hashes & validation_hashes)


# ============================================================================
# Statistics
# ============================================================================

def print_source_distribution(
    name: str,
    rows: list[dict[str, str]],
) -> None:

    counts = Counter(
        row["source"]
        for row in rows
    )

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"Rows: {len(rows):,}")

    for source, count in counts.most_common():

        percentage = (
            100.0 * count / max(1, len(rows))
        )

        print(
            f"{source:20s}"
            f"{count:10,d}"
            f" ({percentage:6.2f}%)"
        )


def categorize(text: str) -> str:

    text = text.lower()

    categories = [
        (
            "network_security",
            [
                "network security",
                "tcp",
                "udp",
                "dns",
                "tls",
                "firewall",
                "packet",
            ],
        ),
        (
            "web_security",
            [
                "xss",
                "csrf",
                "idor",
                "ssrf",
                "sql injection",
                "web application",
            ],
        ),
        (
            "identity_security",
            [
                "active directory",
                "kerberos",
                "ldap",
                "oauth",
                "identity",
                "privilege",
            ],
        ),
        (
            "malware_analysis",
            [
                "malware",
                "ransomware",
                "rootkit",
                "reverse engineering",
                "yara",
            ],
        ),
        (
            "cryptography",
            [
                "cryptography",
                "encryption",
                "cipher",
                "hashing",
                "tls",
            ],
        ),
        (
            "secure_coding",
            [
                "secure coding",
                "input validation",
                "sanitization",
                "secure software",
            ],
        ),
        (
            "threat_intelligence",
            [
                "threat intelligence",
                "ioc",
                "indicator of compromise",
                "apt",
            ],
        ),
        (
            "incident_response",
            [
                "incident response",
                "forensics",
                "triage",
                "containment",
            ],
        ),
        (
            "cloud_security",
            [
                "aws",
                "azure",
                "gcp",
                "cloud security",
                "kubernetes",
            ],
        ),
        (
            "offensive_security",
            [
                "penetration test",
                "red team",
                "exploit",
                "rop",
                "payload",
            ],
        ),
        (
            "social_engineering",
            [
                "phishing",
                "social engineering",
                "pretexting",
            ],
        ),
    ]

    for category, terms in categories:

        if any(term in text for term in terms):
            return category

    return "other"


def print_category_distribution(
    name: str,
    rows: list[dict[str, str]],
) -> None:

    counts = Counter(
        categorize(
            row["user"] + "\n" + row["assistant"]
        )
        for row in rows
    )

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"Rows: {len(rows):,}")

    for category, count in counts.most_common():

        percentage = (
            100.0 * count / max(1, len(rows))
        )

        print(
            f"{category:25s}"
            f"{count:10,d}"
            f" ({percentage:6.2f}%)"
        )


# ============================================================================
# JSONL writer
# ============================================================================

def write_jsonl(
    path: Path,
    rows: list[dict[str, str]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Combine ShareGPT train + validation with Trendyol, "
            "deduplicate, and create a fresh train/validation split."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="data/final",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.02,
        help=(
            "Fraction of the final combined dataset used for validation."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("CYBERSECURITY DATASET MERGE + SPLIT")
    print("=" * 80)

    print("\nINPUTS")
    print(
        f"ShareGPT train      : {SHAREGPT_TRAIN_PATH}"
    )
    print(
        f"ShareGPT validation : {SHAREGPT_VALIDATION_PATH}"
    )
    print(
        f"Trendyol            : {TRENDYOL_DATASET}"
    )

    # ------------------------------------------------------------------
    # 1. Load ShareGPT train
    # ------------------------------------------------------------------

    sharegpt_train, sharegpt_train_stats = (
        load_sharegpt_jsonl(
            SHAREGPT_TRAIN_PATH,
            "train",
        )
    )

    # ------------------------------------------------------------------
    # 2. Load ShareGPT validation
    # ------------------------------------------------------------------

    sharegpt_validation, sharegpt_validation_stats = (
        load_sharegpt_jsonl(
            SHAREGPT_VALIDATION_PATH,
            "validation",
        )
    )

    # ------------------------------------------------------------------
    # 3. Combine ShareGPT train + validation
    # ------------------------------------------------------------------

    sharegpt_all = (
        sharegpt_train
        + sharegpt_validation
    )

    print()
    print("=" * 80)
    print("COMBINED SHAREGPT")
    print("=" * 80)

    print(
        f"ShareGPT train      : {len(sharegpt_train):,}"
    )

    print(
        f"ShareGPT validation : {len(sharegpt_validation):,}"
    )

    print(
        f"ShareGPT total      : {len(sharegpt_all):,}"
    )

    # ------------------------------------------------------------------
    # 4. Load Trendyol
    # ------------------------------------------------------------------

    trendyol, trendyol_stats = load_trendyol()

    # ------------------------------------------------------------------
    # 5. Combine ShareGPT + Trendyol
    # ------------------------------------------------------------------

    merged = (
        sharegpt_all
        + trendyol
    )

    merged_before_dedup = len(merged)

    print()
    print("=" * 80)
    print("COMBINED DATASET BEFORE DEDUPLICATION")
    print("=" * 80)

    print(
        f"ShareGPT : {len(sharegpt_all):,}"
    )

    print(
        f"Trendyol : {len(trendyol):,}"
    )

    print(
        f"Total    : {len(merged):,}"
    )

    # ------------------------------------------------------------------
    # 6. Deduplicate entire combined pool
    # ------------------------------------------------------------------

    merged, duplicates_removed = (
        deduplicate_rows(merged)
    )

    print()
    print("=" * 80)
    print("DEDUPLICATION")
    print("=" * 80)

    print(
        f"Before : {merged_before_dedup:,}"
    )

    print(
        f"Removed: {duplicates_removed:,}"
    )

    print(
        f"After  : {len(merged):,}"
    )

    # ------------------------------------------------------------------
    # 7. Fresh train/validation split
    # ------------------------------------------------------------------

    train, validation = split_dataset(
        merged,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 8. Verify no leakage
    # ------------------------------------------------------------------

    overlap_count = verify_no_overlap(
        train,
        validation,
    )

    if overlap_count != 0:
        raise RuntimeError(
            "Train/validation leakage detected: "
            f"{overlap_count} overlapping examples."
        )

    print()
    print("=" * 80)
    print("FINAL SPLIT")
    print("=" * 80)

    print(
        f"Total      : {len(merged):,}"
    )

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print(
        f"Ratio      : {args.validation_ratio:.4f}"
    )

    print(
        "Leakage    : 0"
    )

    # ------------------------------------------------------------------
    # 9. Statistics
    # ------------------------------------------------------------------

    print_source_distribution(
        "FINAL TRAIN SOURCE DISTRIBUTION",
        train,
    )

    print_source_distribution(
        "FINAL VALIDATION SOURCE DISTRIBUTION",
        validation,
    )

    print_category_distribution(
        "FINAL TRAIN CATEGORY DISTRIBUTION",
        train,
    )

    print_category_distribution(
        "FINAL VALIDATION CATEGORY DISTRIBUTION",
        validation,
    )

    # ------------------------------------------------------------------
    # 10. Output paths
    # ------------------------------------------------------------------

    train_path = (
        output_dir
        / "train.jsonl"
    )

    validation_path = (
        output_dir
        / "validation.jsonl"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    # ------------------------------------------------------------------
    # 11. Write datasets
    # ------------------------------------------------------------------

    write_jsonl(
        train_path,
        train,
    )

    write_jsonl(
        validation_path,
        validation,
    )

    # ------------------------------------------------------------------
    # 12. Metadata
    # ------------------------------------------------------------------

    metadata = {
        "seed": args.seed,

        "validation_ratio": (
            args.validation_ratio
        ),

        "inputs": {
            "sharegpt_train": str(
                SHAREGPT_TRAIN_PATH
            ),
            "sharegpt_validation": str(
                SHAREGPT_VALIDATION_PATH
            ),
            "trendyol": TRENDYOL_DATASET,
        },

        "sharegpt": {
            "train_input_rows": (
                sharegpt_train_stats.get(
                    "input",
                    0,
                )
            ),
            "train_kept_rows": len(
                sharegpt_train
            ),
            "validation_input_rows": (
                sharegpt_validation_stats.get(
                    "input",
                    0,
                )
            ),
            "validation_kept_rows": len(
                sharegpt_validation
            ),
            "combined_rows": len(
                sharegpt_all
            ),
        },

        "trendyol": {
            "input_rows": (
                trendyol_stats.get(
                    "input",
                    0,
                )
            ),
            "kept_rows": len(
                trendyol
            ),
        },

        "merge": {
            "rows_before_dedup": (
                merged_before_dedup
            ),
            "duplicates_removed": (
                duplicates_removed
            ),
            "rows_after_dedup": len(
                merged
            ),
        },

        "final_split": {
            "train_rows": len(
                train
            ),
            "validation_rows": len(
                validation
            ),
            "validation_ratio": (
                args.validation_ratio
            ),
            "train_validation_overlap": (
                overlap_count
            ),
        },

        "processing": {
            "sharegpt_train_and_validation_combined": True,
            "sharegpt_recleaned": False,
            "trendyol_content_filtering": False,
            "generic_programming_filtering": False,
            "offensive_security_filtering": False,
            "structural_validation": True,
            "exact_deduplication": True,
            "fresh_final_split": True,
            "final_shuffle": True,
        },
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # 13. Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)

    print(
        f"Train      : {train_path}"
    )

    print(
        f"Validation : {validation_path}"
    )

    print(
        f"Metadata   : {metadata_path}"
    )

    print()
    print(
        f"Final rows : {len(merged):,}"
    )

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()