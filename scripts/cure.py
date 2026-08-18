#!/usr/bin/env python3

"""
Cybersecurity ShareGPT Dataset Filter

Input:
    data/sharegpt_clean/train.jsonl
    data/sharegpt_clean/validation.jsonl

Output:
    data/sharegpt_filtered/train.jsonl
    data/sharegpt_filtered/validation.jsonl
    data/sharegpt_filtered/train_rejected.jsonl
    data/sharegpt_filtered/validation_rejected.jsonl
    data/sharegpt_filtered/filter_report.json

Filtering:
    - invalid JSON
    - invalid schema
    - empty user/assistant
    - ShareGPT artifacts
    - meta prompts
    - garbled responses
    - code-language mismatch
    - very low question/answer relevance
"""

import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

TRAIN_INPUT = Path("data/sharegpt_clean/train.jsonl")
VAL_INPUT = Path("data/sharegpt_clean/validation.jsonl")

OUTPUT_DIR = Path("data/sharegpt_filtered")

TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
VAL_OUTPUT = OUTPUT_DIR / "validation.jsonl"

TRAIN_REJECTED = OUTPUT_DIR / "train_rejected.jsonl"
VAL_REJECTED = OUTPUT_DIR / "validation_rejected.jsonl"

REPORT_OUTPUT = OUTPUT_DIR / "filter_report.json"


# ============================================================
# CONFIGURATION
# ============================================================

MIN_USER_LENGTH = 50
MIN_ASSISTANT_LENGTH = 80

MAX_USER_LENGTH = 15000
MAX_ASSISTANT_LENGTH = 20000

MIN_RELEVANCE_SCORE = 1


# ============================================================
# REGEX
# ============================================================

CODE_FENCE_RE = re.compile(
    r"```([a-zA-Z0-9_+-]*)\s*(.*?)```",
    re.DOTALL,
)

THINK_RE = re.compile(
    r"<think>.*?</think>",
    re.IGNORECASE | re.DOTALL,
)


# ============================================================
# LANGUAGE INDICATORS
# ============================================================

LANGUAGE_PATTERNS = {
    "python": [
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bprint\s*\(",
        r"\bpython\b",
    ],

    "bash": [
        r"#!/bin/bash",
        r"\bsudo\s+",
        r"\bapt(-get)?\s+",
        r"\bcurl\s+",
        r"\bwget\s+",
        r"\bchmod\s+",
        r"\bsystemctl\s+",
    ],

    "shell": [
        r"#!/bin/sh",
        r"\becho\s+",
        r"\bexport\s+\w+=",
        r"\bif\s+\[",
        r"\bfi\b",
    ],

    "powershell": [
        r"\bGet-\w+",
        r"\bSet-\w+",
        r"\bNew-\w+",
        r"\bInvoke-\w+",
        r"\bRemove-\w+",
        r"\$env:",
        r"\$[A-Za-z_][A-Za-z0-9_]*",
    ],

    "javascript": [
        r"\bconst\s+\w+\s*=",
        r"\blet\s+\w+\s*=",
        r"\bvar\s+\w+\s*=",
        r"\bfunction\s+\w+\s*\(",
        r"\bconsole\.log\s*\(",
        r"=>",
    ],

    "typescript": [
        r"\binterface\s+\w+",
        r"\btype\s+\w+\s*=",
        r":\s*(string|number|boolean)\b",
        r"\bpublic\s+\w+",
        r"\bprivate\s+\w+",
    ],

    "sql": [
        r"\bSELECT\b",
        r"\bINSERT\s+INTO\b",
        r"\bUPDATE\s+\w+\s+SET\b",
        r"\bDELETE\s+FROM\b",
        r"\bCREATE\s+TABLE\b",
        r"\bDROP\s+TABLE\b",
    ],

    "c": [
        r"#include\s*<stdio\.h>",
        r"#include\s*<stdlib\.h>",
        r"\bint\s+main\s*\(",
        r"\bprintf\s*\(",
    ],

    "cpp": [
        r"#include\s*<iostream>",
        r"\bstd::",
        r"\bcout\s*<<",
        r"\bint\s+main\s*\(",
    ],

    "java": [
        r"\bpublic\s+class\s+\w+",
        r"\bpublic\s+static\s+void\s+main",
        r"\bSystem\.out\.println",
    ],

    "go": [
        r"\bpackage\s+main\b",
        r"\bfunc\s+main\s*\(",
        r"\bfmt\.Print",
    ],

    "rust": [
        r"\bfn\s+main\s*\(",
        r"\blet\s+mut\b",
        r"\bprintln!\s*\(",
    ],
}


# ============================================================
# META PROMPT DETECTION
# ============================================================

META_PATTERNS = [
    r"\bignore\s+(all|any|the)\s+previous\s+instructions\b",
    r"\bignore\s+previous\s+instructions\b",
    r"\bdisregard\s+(all|any|the)\s+previous\b",
    r"\bforget\s+(all|any|the)\s+previous\b",
    r"\byou\s+are\s+now\s+(a|an)\b",
    r"\bact\s+as\s+(a|an)\b",
    r"\bpretend\s+you\s+are\b",
    r"\bsystem\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\breveal\s+your\s+instructions\b",
    r"\breveal\s+your\s+prompt\b",
    r"\bshow\s+me\s+your\s+hidden\s+instructions\b",
]


# ============================================================
# SHAREGPT ARTIFACTS
# ============================================================

SHAREGPT_ARTIFACT_PATTERNS = [
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|assistant\|>",
    r"<\|user\|>",
    r"<\|system\|>",
    r"<\|endoftext\|>",
    r"<\|end\|>",
    r"<\|begin_of_text\|>",
    r"<\|eot_id\|>",
    r"<\|start_header_id\|>",
    r"<\|end_header_id\|>",
]


# ============================================================
# GARBLED RESPONSE DETECTION
# ============================================================

def looks_garbled(text: str) -> bool:
    """
    Detect obviously corrupted / malformed responses.

    This intentionally uses conservative thresholds so that
    legitimate cybersecurity code is not unnecessarily removed.
    """

    if not text:
        return True

    # Excessive replacement characters
    if text.count("\ufffd") > 3:
        return True

    # Excessively repeated characters
    if re.search(r"(.)\1{15,}", text):
        return True

    # Excessive escaped garbage
    escaped_count = len(re.findall(r"\\[A-Za-z]", text))

    if escaped_count > 100 and len(text) < 1000:
        return True

    # Very little alphanumeric content
    alphanumeric = len(re.findall(r"[A-Za-z0-9]", text))

    if len(text) > 200 and alphanumeric / len(text) < 0.15:
        return True

    return False


# ============================================================
# CODE EXTRACTION
# ============================================================

def extract_code_blocks(text: str):
    """
    Return:

        [
            {
                "language": "...",
                "code": "..."
            }
        ]
    """

    blocks = []

    for match in CODE_FENCE_RE.finditer(text):
        language = match.group(1).strip().lower()
        code = match.group(2).strip()

        blocks.append(
            {
                "language": language,
                "code": code,
            }
        )

    return blocks


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_languages(text: str):
    """
    Detect languages based on code indicators.
    """

    scores = Counter()

    for language, patterns in LANGUAGE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                scores[language] += 1

    return scores


def normalize_language(language: str):
    language = language.lower().strip()

    aliases = {
        "py": "python",
        "python3": "python",

        "sh": "shell",
        "shellscript": "shell",

        "ps": "powershell",
        "ps1": "powershell",

        "js": "javascript",
        "node": "javascript",

        "ts": "typescript",

        "c++": "cpp",
    }

    return aliases.get(language, language)


# ============================================================
# QUESTION LANGUAGE DETECTION
# ============================================================

def detect_requested_languages(user_text: str):
    """
    Try to determine which programming language the user requested.
    """

    text = user_text.lower()

    requested = set()

    explicit_patterns = {
        "python": [
            r"\bpython\b",
            r"\bpython\s+script\b",
            r"\bpython\s+code\b",
        ],

        "bash": [
            r"\bbash\b",
            r"\bbash\s+script\b",
        ],

        "shell": [
            r"\bshell\s+script\b",
            r"\bsh\s+script\b",
        ],

        "powershell": [
            r"\bpowershell\b",
            r"\bpower\s*shell\b",
        ],

        "javascript": [
            r"\bjavascript\b",
            r"\bnode\.?js\b",
        ],

        "typescript": [
            r"\btypescript\b",
        ],

        "sql": [
            r"\bsql\b",
            r"\bsql\s+query\b",
        ],

        "c": [
            r"\bc\s+program\b",
            r"\bc\s+code\b",
        ],

        "cpp": [
            r"\bc\+\+\b",
            r"\bcpp\b",
        ],

        "java": [
            r"\bjava\b",
        ],

        "go": [
            r"\bgolang\b",
        ],

        "rust": [
            r"\brust\b",
        ],
    }

    for language, patterns in explicit_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text):
                requested.add(language)
                break

    return requested


# ============================================================
# CODE LANGUAGE VALIDATION
# ============================================================

def code_language_mismatch(user: str, assistant: str) -> bool:
    """
    Reject only when there is strong evidence that:

        user explicitly requests language X

    but:

        assistant provides fenced code in a different language.

    If language cannot be confidently determined, keep the row.
    """

    blocks = extract_code_blocks(assistant)

    if not blocks:
        return False

    requested = detect_requested_languages(user)

    if not requested:
        return False

    fenced_languages = {
        normalize_language(block["language"])
        for block in blocks
        if block["language"]
    }

    # No explicit language in the fence -> do not reject.
    if not fenced_languages:
        return False

    # If any requested language is present, it is valid.
    if requested & fenced_languages:
        return False

    # Determine language from code itself.
    detected_scores = detect_languages(
        "\n".join(block["code"] for block in blocks)
    )

    if not detected_scores:
        return False

    detected_language = detected_scores.most_common(1)[0][0]

    if detected_language in requested:
        return False

    return True


# ============================================================
# META PROMPT CHECK
# ============================================================

def contains_meta_prompt(text: str) -> bool:

    for pattern in META_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


# ============================================================
# SHAREGPT ARTIFACT CHECK
# ============================================================

def contains_sharegpt_artifact(text: str):

    for pattern in SHAREGPT_ARTIFACT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


# ============================================================
# RELEVANCE CHECK
# ============================================================

QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "can",
    "could",
    "should",
    "would",
    "explain",
    "describe",
    "provide",
    "write",
    "create",
    "show",
    "give",
    "implement",
    "analyze",
    "design",
    "demonstrate",
}


def relevance_score(user: str, assistant: str):
    """
    Very lightweight lexical relevance check.

    We intentionally avoid aggressive semantic filtering because
    cybersecurity questions can legitimately have answers that
    use different terminology.
    """

    user_tokens = set(
        re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", user.lower())
    )

    assistant_tokens = set(
        re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", assistant.lower())
    )

    if not user_tokens or not assistant_tokens:
        return 0

    overlap = user_tokens & assistant_tokens

    return len(overlap)


def low_relevance(user: str, assistant: str):
    score = relevance_score(user, assistant)

    return score < MIN_RELEVANCE_SCORE


# ============================================================
# ROW VALIDATION
# ============================================================

def validate_row(row):

    if not isinstance(row, dict):
        return "schema_error"

    required = {"system", "user", "assistant", "source"}

    if not required.issubset(row.keys()):
        return "schema_error"

    user = row.get("user", "")
    assistant = row.get("assistant", "")

    if not isinstance(user, str) or not isinstance(assistant, str):
        return "schema_error"

    user = user.strip()
    assistant = assistant.strip()

    if not user:
        return "empty_user"

    if not assistant:
        return "empty_assistant"

    if len(user) < MIN_USER_LENGTH:
        return "short_question"

    if len(assistant) < MIN_ASSISTANT_LENGTH:
        return "short_answer"

    if len(user) > MAX_USER_LENGTH:
        return "question_too_long"

    if len(assistant) > MAX_ASSISTANT_LENGTH:
        return "answer_too_long"

    # --------------------------------------------------------
    # META PROMPT
    # --------------------------------------------------------

    combined = user + "\n" + assistant

    if contains_meta_prompt(combined):
        return "meta_prompt"

    # --------------------------------------------------------
    # SHAREGPT ARTIFACTS
    # --------------------------------------------------------

    if contains_sharegpt_artifact(combined):
        return "sharegpt_artifact"

    # --------------------------------------------------------
    # THINK TAG
    # --------------------------------------------------------

    if THINK_RE.search(combined):
        return "think_contamination"

    # --------------------------------------------------------
    # GARBLED RESPONSE
    # --------------------------------------------------------

    if looks_garbled(assistant):
        return "garbled_response"

    # --------------------------------------------------------
    # LANGUAGE MISMATCH
    # --------------------------------------------------------

    if code_language_mismatch(user, assistant):
        return "code_language_mismatch"

    # --------------------------------------------------------
    # RELEVANCE
    # --------------------------------------------------------

    if low_relevance(user, assistant):
        return "low_question_answer_relevance"

    return None


# ============================================================
# PROCESS FILE
# ============================================================

def process_file(input_path, output_path, rejected_path):

    print("=" * 80)
    print(f"PROCESSING: {input_path}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kept = []
    rejected = []

    invalid_json = 0
    reasons = Counter()

    total = 0

    with input_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            total += 1

            # ------------------------------------------------
            # JSON
            # ------------------------------------------------

            try:
                row = json.loads(line)

            except json.JSONDecodeError as e:

                invalid_json += 1

                rejected_row = {
                    "_filter_reason": "invalid_json",
                    "_line": line_number,
                    "_error": str(e),
                    "_raw": line,
                }

                rejected.append(rejected_row)
                reasons["invalid_json"] += 1

                continue

            # ------------------------------------------------
            # VALIDATE
            # ------------------------------------------------

            reason = validate_row(row)

            if reason is None:

                kept.append(row)

            else:

                reasons[reason] += 1

                row_copy = dict(row)

                row_copy["_filter_reason"] = reason
                row_copy["_line"] = line_number

                rejected.append(row_copy)

    # ========================================================
    # WRITE KEPT
    # ========================================================

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in kept:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # WRITE REJECTED
    # ========================================================

    with rejected_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rejected:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # REPORT
    # ========================================================

    print(f"Input rows       : {total}")
    print(f"Invalid JSON     : {invalid_json}")
    print()

    print("RESULT")
    print("-" * 80)

    print(f"Kept             : {len(kept)}")
    print(f"Rejected         : {len(rejected)}")

    print()

    print("REJECTION REASONS")
    print("-" * 80)

    if reasons:

        for reason, count in reasons.most_common():

            percentage = (
                count / total * 100
                if total
                else 0
            )

            print(
                f"{reason:<32}"
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

    else:

        print("None")

    print()

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rejected_output": str(rejected_path),
        "input_rows": total,
        "kept": len(kept),
        "rejected": len(rejected),
        "invalid_json": invalid_json,
        "rejection_reasons": dict(reasons),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CYBERSECURITY SFT DATASET FILTER")
    print("=" * 80)
    print()

    train_report = process_file(
        TRAIN_INPUT,
        TRAIN_OUTPUT,
        TRAIN_REJECTED,
    )

    print()

    val_report = process_file(
        VAL_INPUT,
        VAL_OUTPUT,
        VAL_REJECTED,
    )

    # ========================================================
    # SAVE GLOBAL REPORT
    # ========================================================

    report = {
        "train": train_report,
        "validation": val_report,
        "configuration": {
            "min_user_length": MIN_USER_LENGTH,
            "min_assistant_length": MIN_ASSISTANT_LENGTH,
            "max_user_length": MAX_USER_LENGTH,
            "max_assistant_length": MAX_ASSISTANT_LENGTH,
            "min_relevance_score": MIN_RELEVANCE_SCORE,
        },
    }

    with REPORT_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Train: "
        f"{train_report['input_rows']} "
        f"→ "
        f"{train_report['kept']}"
    )

    print(
        f"Validation: "
        f"{val_report['input_rows']} "
        f"→ "
        f"{val_report['kept']}"
    )

    print()
    print("OUTPUT")
    print("-" * 80)

    print(f"Train       : {TRAIN_OUTPUT}")
    print(f"Validation  : {VAL_OUTPUT}")
    print(f"Rejected    : {TRAIN_REJECTED}")
    print(f"Rejected    : {VAL_REJECTED}")
    print(f"Report      : {REPORT_OUTPUT}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()