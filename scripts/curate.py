#!/usr/bin/env python3
"""
Curate and merge Trendyol + ShareGPT cybersecurity datasets for LoRA SFT.

Design goals
------------
1. Preserve broad cybersecurity diversity, including offensive-security concepts.
2. Remove structural/data-quality problems.
3. Remove obvious generic-programming noise.
4. Remove rows whose primary purpose is operationally enabling attack/evasion,
   credential theft, persistence, phishing, or exfiltration.
5. Remove the huge ShareGPT meta-system prompt so the model does not learn
   "always output Python / reveal reasoning" behavior.
6. Normalize both sources to:
       {"system": ..., "user": ..., "assistant": ...}
7. Deduplicate on normalized user+assistant text.
8. Produce a train/validation split with reproducible shuffling.

This is intentionally NOT a blanket "offensive content" filter. Conceptual
offensive-security, vulnerability analysis, exploit explanations, secure
coding, detection, and defensive material are retained when they are useful
training examples.
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

from datasets import load_dataset, Dataset, DatasetDict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRENDYOL = "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
SHAREGPT = "ChaoticNeutrals/Cybersecurity-ShareGPT"

SHAREGPT_META_PROMPT_RE = re.compile(
    r"Answer the Question in a logical, step-by-step manner.*?"
    r"Your answer should only return Python code.*?",
    re.I | re.S,
)

GENERIC_PROGRAMMING_PATTERNS = [
    r"\bhello world\b",
    r"\bsort (?:a|an)? ?(?:list|array)\b",
    r"\bcalculator\b",
    r"\bhangman\b",
    r"\btic[- ]?tac[- ]?toe\b",
    r"\bweb scraper\b",
    r"\bweather app\b",
    r"\bto[- ]?do app\b",
    r"\bflask app\b",
    r"\bdjango app\b",
    r"\brest api\b",
    r"\bsimple chatbot\b",
]

# These patterns are deliberately narrow. They target examples whose requested
# capability is directly operational rather than deleting an entire topic.
OPERATIONAL_RISK_PATTERNS = [
    # Credential theft / credential material
    r"\bcredential dumping\b",
    r"\bsteal(?:ing)? credentials\b",
    r"\bextract stored (?:windows )?credentials\b",
    r"\bpassword dumping\b",
    r"\bpassword hashes?\b.*\bcrack\b",
    r"\bpass[- ]the[- ]hash\b",

    # Persistence / malware deployment
    r"\bcreate\b.*\brootkit\b",
    r"\bmalware\b.*\bpersistence\b",
    r"\bpersist(?:ence|ent)\b.*\bregistry\b",
    r"\bpolymorphic\b.*\b(evade|bypass)\b",
    r"\bsandbox evasion\b",

    # Evasion
    r"\bevad(?:e|ing|es)\b.*\b(?:ids|ips|edr|av|antivirus|detection)\b",
    r"\bavoid detection\b",
    r"\bstealthy\b.*\bscan\b",
    r"\bblend in\b.*\btraffic\b",

    # Phishing / social engineering execution
    r"\bspear[- ]phish(?:ing)?\b.*\bsend\b",
    r"\bphishing email\b.*\b(?:send|smtp)\b",
    r"\btrack(?:ing)? pixel\b.*\bphish",
    r"\bcredential harvesting\b",

    # Exfiltration / covert channels
    r"\bdata exfiltration\b.*\b(?:dns|steganography|covert channel)\b",
    r"\bdns tunneling\b.*\b(?:exfiltrat|transmit stolen)\b",
    r"\bcovert channel\b.*\bexfiltrat",

    # Reverse shells / direct payload delivery
    r"\breverse shell\b.*\b(?:connect|payload|script)\b",
    r"\bdownload(?:er)?\b.*\bexecute\b.*\bpayload\b",
]

# Keep these concepts even when the row discusses offensive security.
DEFENSIVE_OVERRIDE_PATTERNS = [
    r"\bhow to detect\b",
    r"\bdetection\b",
    r"\bmitre attack\b",
    r"\bnist\b",
    r"\bsiem\b",
    r"\bedr\b",
    r"\bincident response\b",
    r"\bremediation\b",
    r"\bhardening\b",
    r"\bsecure coding\b",
    r"\bmitigation\b",
    r"\bdefensive\b",
    r"\bblue team\b",
    r"\bthreat hunting\b",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def pair_hash(user: str, assistant: str) -> str:
    payload = normalize_for_hash(user) + "\n<SEP>\n" + normalize_for_hash(assistant)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.I | re.S) for p in patterns)


def looks_generic_programming(user: str, assistant: str) -> bool:
    text = f"{user}\n{assistant}".lower()
    if not contains_any(text, GENERIC_PROGRAMMING_PATTERNS):
        return False

    cyber_terms = [
        "security", "cyber", "vulnerability", "attack", "malware",
        "threat", "cve", "cwe", "authentication", "authorization",
        "encryption", "tls", "network security", "incident", "siem",
        "firewall", "ids", "ips", "edr", "secure coding", "exploit",
    ]
    return not any(term in text for term in cyber_terms)


def operational_risk(user: str, assistant: str) -> bool:
    user_text = clean_text(user)
    assistant_text = clean_text(assistant)
    text = f"{user_text}\n{assistant_text}"

    if not contains_any(text, OPERATIONAL_RISK_PATTERNS):
        return False

    # Only retain a risky-looking example when the user's actual request
    # is clearly defensive/analytical rather than asking to perform the attack.
    defensive_user = contains_any(user_text, DEFENSIVE_OVERRIDE_PATTERNS)

    # Strong operational indicators should never be rescued merely because
    # defensive terminology appears somewhere in the prompt.
    strong_operational_patterns = [
        r"\bsteal(?:ing)? credentials\b",
        r"\bcredential dumping\b",
        r"\bpass[- ]the[- ]hash\b",
        r"\breverse shell\b",
        r"\brootkit\b",
        r"\bcredential harvesting\b",
        r"\bdns tunneling\b",
        r"\bdata exfiltration\b",
        r"\bpolymorphic\b.*\b(?:evade|bypass)\b",
    ]

    if contains_any(text, strong_operational_patterns):
        return True

    return not defensive_user

def strip_sharegpt_system(system: str) -> str:
    """
    Remove the repeated meta-instruction found in this ShareGPT source.
    Preserve any unrelated system content if present.
    """
    system = clean_text(system)
    if not system:
        return ""

    stripped = SHAREGPT_META_PROMPT_RE.sub("", system).strip()
    # The source's system message is essentially the same meta prompt in the
    # inspected examples. Do not carry it into SFT if it leaves no useful
    # domain-specific instruction.
    if len(stripped) < 20:
        return ""
    return stripped


def extract_sharegpt_turns(conversation: Any) -> tuple[str, str, str] | None:
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except json.JSONDecodeError:
            return None

    if not isinstance(conversation, list):
        return None

    system = ""
    user = ""
    assistant = ""

    # We only construct a single-turn SFT example. This avoids accidentally
    # mixing multiple unrelated Q/A pairs in one training record.
    pending_user = None

    for turn in conversation:
        if not isinstance(turn, dict):
            continue

        role = str(turn.get("from", turn.get("role", ""))).lower().strip()
        value = clean_text(turn.get("value", turn.get("content", "")))

        if not value:
            continue

        if role in {"system"} and not system:
            system = value

        elif role in {"human", "user"} and pending_user is None:
            pending_user = value

        elif role in {"gpt", "assistant", "model"} and pending_user is not None:
            user = pending_user
            assistant = value
            break

    if not user or not assistant:
        return None

    return strip_sharegpt_system(system), user, assistant


def convert_trendyol(row: dict[str, Any]) -> tuple[str, str, str] | None:
    system = clean_text(row.get("system"))
    user = clean_text(row.get("user"))
    assistant = clean_text(row.get("assistant"))

    if not user or not assistant:
        return None

    return system, user, assistant


def basic_quality_ok(system: str, user: str, assistant: str) -> bool:
    if not user or not assistant:
        return False

    # Extremely short examples are unlikely to be useful SFT pairs.
    if len(user) < 20:
        return False

    if len(assistant) < 20:
        return False

    # Reject obviously broken records.
    if user.lower() == assistant.lower():
        return False

    # Reject placeholder-only responses.
    placeholder_patterns = [
        r"^\s*pass\s*$",
        r"^\s*n/?a\s*$",
        r"^\s*todo\s*$",
        r"^\s*placeholder\s*$",
        r"^\s*i don't know\.?\s*$",
    ]

    if contains_any(assistant, placeholder_patterns):
        return False

    return True


def categorize(text: str) -> str:
    t = text.lower()

    rules = [
        ("network_security", ["network security", "tcp", "udp", "dns", "tls", "firewall", "packet"]),
        ("web_security", ["xss", "csrf", "idor", "ssrf", "sql injection", "web application"]),
        ("identity_security", ["active directory", "kerberos", "ldap", "oauth", "identity", "privilege"]),
        ("malware_analysis", ["malware", "ransomware", "rootkit", "reverse engineering", "yara"]),
        ("cryptography", ["cryptography", "encryption", "cipher", "hashing", "tls"]),
        ("secure_coding", ["secure coding", "sanitization", "input validation", "secure software"]),
        ("threat_intelligence", ["threat intelligence", "ioc", "indicator of compromise", "apt"]),
        ("incident_response", ["incident response", "forensics", "triage", "containment"]),
        ("cloud_security", ["aws", "azure", "gcp", "cloud security", "kubernetes"]),
        ("offensive_security", ["penetration test", "red team", "exploit", "rop", "payload"]),
        ("social_engineering", ["phishing", "social engineering", "pretexting"]),
    ]

    for category, terms in rules:
        if any(term in t for term in terms):
            return category

    return "other"


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_trendyol(path: str) -> tuple[list[dict[str, Any]], Counter]:
    ds = load_dataset(path)
    if isinstance(ds, DatasetDict):
        ds = ds["train"]

    stats = Counter()
    output = []

    for row in ds:
        stats["input"] += 1

        converted = convert_trendyol(row)
        if converted is None:
            stats["malformed_or_empty"] += 1
            continue

        system, user, assistant = converted

        if not basic_quality_ok(system, user, assistant):
            stats["quality_rejected"] += 1
            continue

        if looks_generic_programming(user, assistant):
            stats["generic_programming_rejected"] += 1
            continue

        if operational_risk(user, assistant):
            stats["operational_risk_rejected"] += 1
            continue

        output.append({
            "system": system,
            "user": user,
            "assistant": assistant,
            "source": "trendyol",
        })
        stats["kept"] += 1

    return output, stats


def process_sharegpt(path: str) -> tuple[list[dict[str, Any]], Counter]:
    ds = load_dataset(path)
    if isinstance(ds, DatasetDict):
        # ShareGPT is commonly stored under train.
        ds = ds["train"]

    stats = Counter()
    output = []

    for row in ds:
        stats["input"] += 1

        converted = extract_sharegpt_turns(row.get("conversations"))
        if converted is None:
            stats["malformed_or_empty"] += 1
            continue

        system, user, assistant = converted

        if not basic_quality_ok(system, user, assistant):
            stats["quality_rejected"] += 1
            continue

        if looks_generic_programming(user, assistant):
            stats["generic_programming_rejected"] += 1
            continue

        if operational_risk(user, assistant):
            stats["operational_risk_rejected"] += 1
            continue

        output.append({
            "system": system,
            "user": user,
            "assistant": assistant,
            "source": "sharegpt",
        })
        stats["kept"] += 1

    return output, stats


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen = set()
    unique = []

    for row in rows:
        key = pair_hash(row["user"], row["assistant"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    return unique, len(rows) - len(unique)


def split_rows(
    rows: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * validation_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats(name: str, rows: list[dict[str, Any]]) -> None:
    counts = Counter(
        categorize(f'{r["user"]}\n{r["assistant"]}')
        for r in rows
    )

    print(f"\n{'=' * 80}")
    print(name)
    print(f"{'=' * 80}")
    print(f"Rows: {len(rows):,}")

    for category, count in counts.most_common():
        pct = 100 * count / max(1, len(rows))
        print(f"{category:25s} {count:7,d} ({pct:6.2f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="data/final",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LOADING + CURATING DATASETS")
    print("=" * 80)

    print("\nLoading Trendyol:")
    print(f"  {TRENDYOL}")

    trendyol, trendyol_stats = process_trendyol(TRENDYOL)

    print("\nLoading ShareGPT:")
    print(f"  {SHAREGPT}")

    sharegpt, sharegpt_stats = process_sharegpt(SHAREGPT)

    print("\nTRENDYOL")
    for k, v in trendyol_stats.items():
        print(f"{k:30s}: {v:,}")

    print("\nSHAREGPT")
    for k, v in sharegpt_stats.items():
        print(f"{k:30s}: {v:,}")

    merged = trendyol + sharegpt

    before = len(merged)
    merged, duplicate_count = deduplicate(merged)

    print("\nMERGE")
    print(f"Before deduplication : {before:,}")
    print(f"Duplicates removed   : {duplicate_count:,}")
    print(f"Final rows           : {len(merged):,}")

    print_stats("FINAL DISTRIBUTION", merged)

    train, validation = split_rows(
        merged,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)
    source_counts = Counter(row["source"] for row in merged)

    metadata = {
    "seed": args.seed,
    "validation_ratio": args.validation_ratio,

    "trendyol_input": TRENDYOL,
    "sharegpt_input": SHAREGPT,

    "trendyol_stats": dict(trendyol_stats),
    "sharegpt_stats": dict(sharegpt_stats),

    "merged_before_dedup": before,
    "duplicates_removed": duplicate_count,
    "final_rows": len(merged),
    "train_rows": len(train),
    "validation_rows": len(validation),

    "policy": {
        "preserve_offensive_security_diversity": True,
        "remove_structural_noise": True,
        "remove_generic_programming_noise": True,
        "remove_direct_operational_attack_examples": True,
        "remove_sharegpt_meta_reasoning_prompt": True,
    },
}

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nOUTPUT")
    print("\nFINAL SOURCE DISTRIBUTION")
    for source, count in source_counts.most_common():
        pct = 100 * count / max(1, len(merged))
        print(f"{source:15s} {count:7,d} ({pct:6.2f}%)")
    print(f"Train      : {output_dir / 'train.jsonl'}")
    print(f"Validation : {output_dir / 'validation.jsonl'}")
    print(f"Metadata   : {output_dir / 'metadata.json'}")
    print("\nDONE")


if __name__ == "__main__":
    main()