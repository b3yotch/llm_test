#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Strong indicators of visible "thinking aloud" / reasoning-style responses.
# We intentionally keep these relatively specific to avoid flagging normal
# explanatory prose.
REASONING_PATTERNS = {
    "okay_i_need_to": re.compile(
        r"\bokay[, ]+(?:so )?i need to\b", re.I
    ),
    "let_me_start": re.compile(
        r"\blet me start\b", re.I
    ),
    "let_me_recall": re.compile(
        r"\blet me (?:recall|remember)\b", re.I
    ),
    "let_me_think": re.compile(
        r"\blet me think\b", re.I
    ),
    "lets_think": re.compile(
        r"\blet'?s think\b", re.I
    ),
    "i_need_to_figure": re.compile(
        r"\bi need to figure (?:out|this)\b", re.I
    ),
    "first_i_need_to": re.compile(
        r"\bfirst[, ]+i need to\b", re.I
    ),
    "i_should_mention": re.compile(
        r"\bi should mention\b", re.I
    ),
    "wait_how": re.compile(
        r"\bwait[, ]+(?:how|what|why)\b", re.I
    ),
    "thinking_about": re.compile(
        r"\b(?:thinking|think) (?:about|through) (?:this|the problem)\b",
        re.I,
    ),
    "i_have_to": re.compile(
        r"\bi have to (?:explain|figure|determine|consider)\b",
        re.I,
    ),
}

# Remnants of the ShareGPT system/meta prompt.
META_PROMPT_PATTERNS = {
    "logical_step_by_step": re.compile(
        r"logical[, ]+step[- ]by[- ]step", re.I
    ),
    "reasoning_process_clear": re.compile(
        r"reasoning process.*clear", re.I | re.S
    ),
    "break_down_issue": re.compile(
        r"break (?:down|the) (?:issue|problem) into", re.I
    ),
    "multiple_hypotheses": re.compile(
        r"multiple hypotheses", re.I
    ),
    "critical_analysis": re.compile(
        r"critically evaluate", re.I
    ),
    "reasoning_chain": re.compile(
        r"reasoning chain", re.I
    ),
    "answer_only_python": re.compile(
        r"answer.*only return python code", re.I | re.S
    ),
}

# Basic signals that the response contains code.
CODE_PATTERNS = [
    re.compile(r"```(?:python|py|bash|shell|sql|javascript|js|powershell)?", re.I),
    re.compile(r"\bimport\s+[A-Za-z_][A-Za-z0-9_.]*"),
    re.compile(r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\("),
    re.compile(r"\bSELECT\s+.+\s+FROM\b", re.I | re.S),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                rows.append(row)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARNING] Invalid JSON at {path}:{line_no}: {exc}"
                )

    return rows


def clean(text):
    if text is None:
        return ""

    return str(text).replace("\x00", " ").strip()


def normalize(text):
    text = clean(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def find_matches(text, patterns):
    matches = []

    for name, pattern in patterns.items():
        if pattern.search(text):
            matches.append(name)

    return matches


def contains_code(text):
    return any(pattern.search(text) for pattern in CODE_PATTERNS)


def analyze_split(name, rows, samples):
    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    print(f"Rows: {len(rows):,}")

    source_counts = Counter(
        clean(row.get("source", "unknown"))
        for row in rows
    )

    print("\nSOURCE DISTRIBUTION")
    for source, count in source_counts.most_common():
        pct = 100 * count / max(1, len(rows))
        print(f"{source:20s} {count:8,d} ({pct:6.2f}%)")

    # -------------------------------------------------------
    # Reasoning-style analysis
    # -------------------------------------------------------

    reasoning_rows = []
    reasoning_counts = Counter()

    for idx, row in enumerate(rows):
        assistant = clean(row.get("assistant", ""))

        matches = find_matches(
            assistant,
            REASONING_PATTERNS,
        )

        if matches:
            reasoning_rows.append(
                {
                    "index": idx,
                    "matches": matches,
                    "user": clean(row.get("user", "")),
                    "assistant": assistant,
                    "source": clean(row.get("source", "")),
                }
            )

            reasoning_counts.update(matches)

    print("\nVISIBLE REASONING-STYLE RESPONSES")
    print(
        f"Rows containing reasoning-style markers: "
        f"{len(reasoning_rows):,} / {len(rows):,} "
        f"({100 * len(reasoning_rows) / max(1, len(rows)):.2f}%)"
    )

    if reasoning_counts:
        print("\nReasoning markers:")
        for pattern, count in reasoning_counts.most_common():
            print(f"  {pattern:25s}: {count:,}")

    # -------------------------------------------------------
    # Meta prompt remnants
    # -------------------------------------------------------

    meta_rows = []

    for idx, row in enumerate(rows):
        text = "\n".join(
            [
                clean(row.get("system", "")),
                clean(row.get("user", "")),
                clean(row.get("assistant", "")),
            ]
        )

        matches = find_matches(
            text,
            META_PROMPT_PATTERNS,
        )

        if matches:
            meta_rows.append(
                {
                    "index": idx,
                    "matches": matches,
                    "user": clean(row.get("user", "")),
                    "assistant": clean(row.get("assistant", "")),
                    "system": clean(row.get("system", "")),
                }
            )

    print("\nSHAREGPT META-PROMPT REMNANTS")
    print(
        f"Rows containing meta-prompt markers: "
        f"{len(meta_rows):,} / {len(rows):,} "
        f"({100 * len(meta_rows) / max(1, len(rows)):.2f}%)"
    )

    # -------------------------------------------------------
    # Code prevalence
    # -------------------------------------------------------

    code_count = 0

    for row in rows:
        if contains_code(clean(row.get("assistant", ""))):
            code_count += 1

    print("\nCODE CONTENT")
    print(
        f"Rows containing code signals: "
        f"{code_count:,} / {len(rows):,} "
        f"({100 * code_count / max(1, len(rows)):.2f}%)"
    )

    # -------------------------------------------------------
    # Length statistics
    # -------------------------------------------------------

    user_lengths = [
        len(clean(row.get("user", "")))
        for row in rows
    ]

    assistant_lengths = [
        len(clean(row.get("assistant", "")))
        for row in rows
    ]

    def stats(values):
        if not values:
            return 0, 0, 0, 0

        values = sorted(values)

        return (
            min(values),
            sum(values) / len(values),
            values[len(values) // 2],
            max(values),
        )

    umin, uavg, umedian, umax = stats(user_lengths)
    amin, aavg, amedian, amax = stats(assistant_lengths)

    print("\nTEXT LENGTHS (characters)")

    print(
        f"User      min={umin:,} "
        f"avg={uavg:,.1f} "
        f"median={umedian:,} "
        f"max={umax:,}"
    )

    print(
        f"Assistant  min={amin:,} "
        f"avg={aavg:,.1f} "
        f"median={amedian:,} "
        f"max={amax:,}"
    )

    # -------------------------------------------------------
    # Print actual suspicious examples
    # -------------------------------------------------------

    print()
    print("-" * 90)
    print("REASONING-STYLE EXAMPLES")
    print("-" * 90)

    for item in reasoning_rows[:samples]:
        print()
        print(f"[INDEX {item['index']}]")
        print(f"SOURCE: {item['source']}")
        print(f"MATCHES: {', '.join(item['matches'])}")

        print("\nUSER:")
        print(item["user"][:1000])

        print("\nASSISTANT:")
        print(item["assistant"][:2000])

    print()
    print("-" * 90)
    print("META-PROMPT EXAMPLES")
    print("-" * 90)

    for item in meta_rows[:samples]:
        print()
        print(f"[INDEX {item['index']}]")
        print(f"MATCHES: {', '.join(item['matches'])}")

        print("\nSYSTEM:")
        print(item["system"][:1500])

        print("\nASSISTANT:")
        print(item["assistant"][:1500])

    return reasoning_rows, meta_rows


# ---------------------------------------------------------------------------
# Train/validation overlap
# ---------------------------------------------------------------------------

def compare_splits(train, validation):
    print()
    print("=" * 90)
    print("TRAIN / VALIDATION LEAKAGE CHECK")
    print("=" * 90)

    train_pairs = set()
    train_users = set()

    for row in train:
        user = normalize(row.get("user", ""))
        assistant = normalize(row.get("assistant", ""))

        train_pairs.add(
            user + "\n<SEP>\n" + assistant
        )

        train_users.add(user)

    validation_pairs = set()
    validation_users = set()

    for row in validation:
        user = normalize(row.get("user", ""))
        assistant = normalize(row.get("assistant", ""))

        validation_pairs.add(
            user + "\n<SEP>\n" + assistant
        )

        validation_users.add(user)

    exact_pair_overlap = train_pairs & validation_pairs
    exact_user_overlap = train_users & validation_users

    print(
        f"Exact user+assistant overlap : "
        f"{len(exact_pair_overlap):,}"
    )

    print(
        f"Exact user overlap           : "
        f"{len(exact_user_overlap):,}"
    )

    if exact_pair_overlap:
        print(
            "\nWARNING: Exact examples occur in both train and validation."
        )
    else:
        print("\nNo exact train/validation pair leakage detected.")

    if exact_user_overlap:
        print(
            "NOTE: Some user prompts occur in both splits with "
            "potentially different answers."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose SFT train/validation datasets."
    )

    parser.add_argument(
        "--train",
        default="data/final/train.jsonl",
    )

    parser.add_argument(
        "--validation",
        default="data/final/validation.jsonl",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of suspicious examples to display.",
    )

    args = parser.parse_args()

    train_path = Path(args.train)
    validation_path = Path(args.validation)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {validation_path}"
        )

    print("=" * 90)
    print("CYBERSECURITY SFT DATASET DIAGNOSIS")
    print("=" * 90)

    print(f"\nTrain file      : {train_path}")
    print(f"Validation file : {validation_path}")

    train = load_jsonl(train_path)
    validation = load_jsonl(validation_path)

    train_reasoning, train_meta = analyze_split(
        "TRAINING SET",
        train,
        args.samples,
    )

    val_reasoning, val_meta = analyze_split(
        "VALIDATION SET",
        validation,
        args.samples,
    )

    compare_splits(train, validation)

    # -------------------------------------------------------
    # Final summary
    # -------------------------------------------------------

    print()
    print("=" * 90)
    print("DIAGNOSIS SUMMARY")
    print("=" * 90)

    train_reasoning_pct = (
        100 * len(train_reasoning) / max(1, len(train))
    )

    val_reasoning_pct = (
        100 * len(val_reasoning) / max(1, len(validation))
    )

    train_meta_pct = (
        100 * len(train_meta) / max(1, len(train))
    )

    val_meta_pct = (
        100 * len(val_meta) / max(1, len(validation))
    )

    print(
        f"""
Visible reasoning-style contamination:

  Train      : {len(train_reasoning):,} "
                 f"({train_reasoning_pct:.2f}%)
  Validation : {len(val_reasoning):,} "
                 f"({val_reasoning_pct:.2f}%)

ShareGPT meta-prompt remnants:

  Train      : {len(train_meta):,} "
                 f"({train_meta_pct:.2f}%)
  Validation : {len(val_meta):,} "
                 f"({val_meta_pct:.2f}%)
"""
    )

    print("\nInterpretation:")
    print(
        "  1. If reasoning-style percentages are high in BOTH splits,"
    )
    print(
        "     the low validation loss is compatible with the model"
    )
    print(
        "     simply learning the dataset's response style."
    )
    print(
        "  2. If reasoning-style examples are mostly in TRAIN,"
    )
    print(
        "     investigate train/validation construction and source mix."
    )
    print(
        "  3. If meta-prompt remnants are present, the curator needs"
    )
    print(
        "     another cleaning pass."
    )
    print(
        "  4. Exact train/validation pair overlap should be zero."
    )
    print()
    print("DONE")


if __name__ == "__main__":
    main()