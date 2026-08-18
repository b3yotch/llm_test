#!/usr/bin/env python3

import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path("data/sharegpt_balanced/train.jsonl")
VAL_PATH = Path("data/sharegpt_balanced/validation.jsonl")


# ============================================================
# REGEX / KEYWORDS
# ============================================================

LANGUAGE_PATTERNS = {
    "python": [
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\bdef\s+\w+\s*\(",
        r"\bprint\s*\(",
        r"\.py\b",
    ],
    "bash": [
        r"#!/bin/bash",
        r"#!/usr/bin/env bash",
        r"\bsudo\s+",
        r"\bapt(-get)?\s+",
        r"\bcurl\s+",
        r"\bwget\s+",
    ],
    "shell": [
        r"#!/bin/sh",
        r"\bchmod\s+",
        r"\bgrep\s+",
        r"\bsed\s+",
        r"\bawk\s+",
    ],
    "powershell": [
        r"\bPowerShell\b",
        r"\bGet-\w+",
        r"\bSet-\w+",
        r"\bNew-\w+",
        r"\bInvoke-\w+",
        r"\$\w+\s*=",
    ],
    "javascript": [
        r"\bconsole\.log\s*\(",
        r"\bconst\s+\w+\s*=",
        r"\blet\s+\w+\s*=",
        r"\bfunction\s+\w+\s*\(",
        r"=>",
    ],
    "typescript": [
        r"\binterface\s+\w+",
        r"\btype\s+\w+\s*=",
        r":\s*(string|number|boolean)\b",
        r"\basync\s+function\b",
    ],
    "sql": [
        r"\bSELECT\b.+\bFROM\b",
        r"\bINSERT\s+INTO\b",
        r"\bUPDATE\b.+\bSET\b",
        r"\bDELETE\s+FROM\b",
        r"\bCREATE\s+TABLE\b",
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
        r"\busing\s+namespace\s+std",
    ],
    "java": [
        r"\bpublic\s+class\s+\w+",
        r"\bSystem\.out\.println",
        r"\bpublic\s+static\s+void\s+main",
    ],
    "go": [
        r"\bpackage\s+main\b",
        r"\bfunc\s+main\s*\(",
        r"\bimport\s+\(",
    ],
    "rust": [
        r"\bfn\s+main\s*\(",
        r"\blet\s+mut\s+",
        r"\buse\s+std::",
    ],
}


CYBER_TOPICS = {
    "network_security": [
        "network security",
        "firewall",
        "tcp",
        "udp",
        "packet",
        "dns",
        "network traffic",
        "ip address",
        "port scanning",
        "intrusion detection",
    ],
    "web_security": [
        "web security",
        "web application",
        "http",
        "https",
        "xss",
        "csrf",
        "idor",
        "ssrf",
        "sql injection",
        "authentication",
        "authorization",
    ],
    "identity_security": [
        "identity",
        "iam",
        "identity and access",
        "authentication",
        "authorization",
        "access control",
        "oauth",
        "saml",
        "jwt",
        "privilege",
    ],
    "cryptography": [
        "cryptography",
        "encryption",
        "decryption",
        "rsa",
        "aes",
        "hash",
        "sha256",
        "certificate",
        "tls",
        "ssl",
        "key exchange",
    ],
    "malware": [
        "malware",
        "trojan",
        "ransomware",
        "virus",
        "rootkit",
        "spyware",
        "malicious software",
    ],
    "vulnerability": [
        "vulnerability",
        "zero-day",
        "zero day",
        "cve",
        "exploit",
        "buffer overflow",
        "injection",
        "misconfiguration",
        "security flaw",
    ],
    "incident_response": [
        "incident response",
        "incident",
        "forensics",
        "containment",
        "eradication",
        "recovery",
        "security incident",
    ],
    "threat_intelligence": [
        "threat intelligence",
        "threat actor",
        "apt",
        "indicator of compromise",
        "ioc",
        "threat hunting",
        "mitre att&ck",
    ],
    "secure_coding": [
        "secure coding",
        "secure development",
        "input validation",
        "sanitization",
        "code review",
        "security best practices",
    ],
}


SECURITY_INTENT = {
    "offensive": [
        "penetration testing",
        "red team",
        "ethical hacking",
        "exploit",
        "payload",
        "attack",
        "attacker",
        "offensive security",
        "post-exploitation",
        "privilege escalation",
        "reverse shell",
        "command and control",
        "c2",
    ],
    "defensive": [
        "defensive",
        "defense",
        "defence",
        "detection",
        "mitigation",
        "monitoring",
        "incident response",
        "threat hunting",
        "security control",
        "hardening",
        "prevention",
        "blue team",
    ],
}


RISKY_BEHAVIOR = {
    "exploit_development": [
        "exploit development",
        "buffer overflow",
        "rop chain",
        "remote code execution",
        "proof of concept exploit",
        "poc exploit",
        "shellcode",
    ],
    "persistence": [
        "persistence",
        "scheduled task",
        "startup folder",
        "registry run key",
        "cron job",
        "service persistence",
    ],
    "credential_access": [
        "credential dumping",
        "password dumping",
        "credential access",
        "keylogger",
        "lsass",
        "password hash",
        "credential theft",
    ],
    "evasion": [
        "evasion",
        "bypass antivirus",
        "bypass av",
        "disable antivirus",
        "disable defender",
        "obfuscation",
        "amsi bypass",
        "edr bypass",
    ],
    "c2": [
        "command and control",
        "c2 server",
        "c2",
        "reverse shell",
        "beacon",
        "callback server",
    ],
    "exfiltration": [
        "exfiltration",
        "exfiltrate",
        "steal data",
        "data theft",
        "upload stolen",
    ],
    "malware_development": [
        "malware development",
        "malware development",
        "ransomware development",
        "trojan development",
        "backdoor",
    ],
}


CONTAMINATION = {
    "meta_prompt": [
        "ignore previous instructions",
        "ignore all previous instructions",
        "system prompt",
        "you are chatgpt",
        "follow these instructions instead",
        "disregard previous",
    ],
    "think": [
        "<think>",
        "</think>",
    ],
    "step-by-step": [
        "step-by-step",
        "step by step",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    rows = []
    invalid = 0

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return rows, invalid

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError:
                invalid += 1

    return rows, invalid


def get_text(row, key):
    value = row.get(key, "")

    if value is None:
        return ""

    if not isinstance(value, str):
        return str(value)

    return value.strip()


def pct(value, total):
    if total == 0:
        return 0.0

    return value / total * 100


def print_counter(counter, total, width=32):
    for key, value in counter.items():
        print(
            f"{key:<{width}} : "
            f"{value:>7} ({pct(value, total):6.2f}%)"
        )


def count_pattern_matches(text, patterns):
    text_lower = text.lower()

    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            return True

    return False


def has_code_fence(text):
    return "```" in text


def response_format(assistant):
    if not assistant:
        return "empty"

    has_code = has_code_fence(assistant)

    if not has_code:
        return "explanation_only"

    # Remove code blocks and see how much explanatory text remains.
    cleaned = re.sub(
        r"```.*?```",
        "",
        assistant,
        flags=re.DOTALL
    ).strip()

    if len(cleaned) < 100:
        return "code_dominant"

    return "explanation_plus_code"


def detect_languages(assistant):
    found = []

    for language, patterns in LANGUAGE_PATTERNS.items():
        if count_pattern_matches(assistant, patterns):
            found.append(language)

    return found


def duplicate_count(values):
    counts = Counter(values)

    return sum(
        count - 1
        for count in counts.values()
        if count > 1
    )


# ============================================================
# DATASET DIAGNOSIS
# ============================================================

def diagnose_dataset(name, path):
    print()
    print("=" * 90)
    print(f"{name.upper()}: {path}")
    print("=" * 90)

    rows, invalid_json = load_jsonl(path)

    print(f"JSON rows loaded       : {len(rows)}")
    print(f"Invalid JSON rows      : {invalid_json}")

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    schema_counter = Counter()
    schema_problems = 0

    users = []
    assistants = []
    sources = []

    for row in rows:
        if not isinstance(row, dict):
            schema_counter["non_dict"] += 1
            schema_problems += 1
            continue

        required = ["system", "user", "assistant", "source"]

        missing = [key for key in required if key not in row]

        if missing:
            schema_counter["missing_keys"] += 1
            schema_problems += 1
            continue

        if not isinstance(row["user"], str):
            schema_counter["user_not_string"] += 1
            schema_problems += 1

        if not isinstance(row["assistant"], str):
            schema_counter["assistant_not_string"] += 1
            schema_problems += 1

        users.append(get_text(row, "user"))
        assistants.append(get_text(row, "assistant"))
        sources.append(get_text(row, "source"))

    print()
    print("SCHEMA")
    print("-" * 90)

    if not schema_counter:
        print("flat                     : "
              f"{len(rows)}")

    else:
        print_counter(schema_counter, len(rows))

    print(
        f"\nRows with schema problems : {schema_problems}"
    )

    # --------------------------------------------------------
    # Lengths
    # --------------------------------------------------------

    user_lengths = [len(x) for x in users]
    assistant_lengths = [len(x) for x in assistants]

    print()
    print("LENGTHS")
    print("-" * 90)

    if user_lengths:
        print(
            f"User character length      : "
            f"{min(user_lengths)} / "
            f"{max(user_lengths)} / "
            f"{sum(user_lengths)/len(user_lengths):.1f}"
        )

        print(
            f"Assistant character length : "
            f"{min(assistant_lengths)} / "
            f"{max(assistant_lengths)} / "
            f"{sum(assistant_lengths)/len(assistant_lengths):.1f}"
        )

    empty_users = sum(not x for x in users)
    empty_assistants = sum(not x for x in assistants)

    print(f"Empty users                : {empty_users}")
    print(f"Empty assistants          : {empty_assistants}")

    # --------------------------------------------------------
    # Response format
    # --------------------------------------------------------

    formats = Counter(
        response_format(a)
        for a in assistants
    )

    print()
    print("RESPONSE FORMAT")
    print("-" * 90)

    for key in [
        "explanation_only",
        "explanation_plus_code",
        "code_dominant",
        "empty",
    ]:
        value = formats.get(key, 0)

        print(
            f"{key:<30} : "
            f"{value:>7} ({pct(value, len(rows)):6.2f}%)"
        )

    # --------------------------------------------------------
    # Languages
    # --------------------------------------------------------

    language_counts = Counter()

    for assistant in assistants:
        for language in detect_languages(assistant):
            language_counts[language] += 1

    print()
    print("CODE / LANGUAGE INDICATORS")
    print("-" * 90)

    for language in LANGUAGE_PATTERNS:
        value = language_counts.get(language, 0)

        print(
            f"{language:<30} : "
            f"{value:>7} ({pct(value, len(rows)):6.2f}%)"
        )

    # --------------------------------------------------------
    # Cybersecurity topics
    # --------------------------------------------------------

    topic_counts = Counter()

    for user, assistant in zip(users, assistants):
        text = user + "\n" + assistant

        for topic, keywords in CYBER_TOPICS.items():
            if any(k.lower() in text.lower() for k in keywords):
                topic_counts[topic] += 1

    print()
    print("CYBERSECURITY TOPICS")
    print("-" * 90)

    for topic in CYBER_TOPICS:
        value = topic_counts.get(topic, 0)

        print(
            f"{topic:<30} : "
            f"{value:>7} ({pct(value, len(rows)):6.2f}%)"
        )

    # --------------------------------------------------------
    # Security intent
    # --------------------------------------------------------

    intent_counts = Counter()

    for user, assistant in zip(users, assistants):
        text = user + "\n" + assistant

        for intent, keywords in SECURITY_INTENT.items():
            if any(k.lower() in text.lower() for k in keywords):
                intent_counts[intent] += 1

    print()
    print("SECURITY INTENT")
    print("-" * 90)

    for intent in SECURITY_INTENT:
        value = intent_counts.get(intent, 0)

        print(
            f"{intent:<30} : "
            f"{value:>7} ({pct(value, len(rows)):6.2f}%)"
        )

    # --------------------------------------------------------
    # Risky behavior
    # --------------------------------------------------------

    risky_counts = Counter()

    for user, assistant in zip(users, assistants):
        text = user + "\n" + assistant

        for category, keywords in RISKY_BEHAVIOR.items():
            if any(k.lower() in text.lower() for k in keywords):
                risky_counts[category] += 1

    print()
    print("OFFENSIVE / RISKY BEHAVIOR")
    print("-" * 90)

    for category in RISKY_BEHAVIOR:
        value = risky_counts.get(category, 0)

        print(
            f"{category:<30} : "
            f"{value:>7} ({pct(value, len(rows)):6.2f}%)"
        )

    # --------------------------------------------------------
    # Contamination
    # --------------------------------------------------------

    contamination_counts = Counter()

    for user, assistant in zip(users, assistants):
        text = user + "\n" + assistant

        for category, keywords in CONTAMINATION.items():
            if any(k.lower() in text.lower() for k in keywords):
                contamination_counts[category] += 1

    print()
    print("CONTAMINATION")
    print("-" * 90)

    for category in CONTAMINATION:
        value = contamination_counts.get(category, 0)

        print(
            f"{category:<30} : "
            f"{value:>7} ({pct(value, len(rows)):6.2f}%)"
        )

    # --------------------------------------------------------
    # Code blocks
    # --------------------------------------------------------

    code_fences = sum(
        has_code_fence(a)
        for a in assistants
    )

    print()
    print("CODE")
    print("-" * 90)

    print(
        f"Rows containing code fences   : "
        f"{code_fences:>7} "
        f"({pct(code_fences, len(rows)):6.2f}%)"
    )

    # --------------------------------------------------------
    # Answer length
    # --------------------------------------------------------

    very_short = sum(
        0 < len(a) < 500
        for a in assistants
    )

    very_long = sum(
        len(a) > 6000
        for a in assistants
    )

    print()
    print("ANSWER LENGTH DISTRIBUTION")
    print("-" * 90)

    print(
        f"Very short (<500 chars)       : "
        f"{very_short:>7} "
        f"({pct(very_short, len(rows)):6.2f}%)"
    )

    print(
        f"Very long (>6000 chars)       : "
        f"{very_long:>7} "
        f"({pct(very_long, len(rows)):6.2f}%)"
    )

    # --------------------------------------------------------
    # Duplication
    # --------------------------------------------------------

    user_duplicates = duplicate_count(users)
    assistant_duplicates = duplicate_count(assistants)

    pairs = [
        user + "\n" + assistant
        for user, assistant in zip(users, assistants)
    ]

    pair_duplicates = duplicate_count(pairs)

    print()
    print("DUPLICATION")
    print("-" * 90)

    print(f"Duplicate user prompts     : {user_duplicates}")
    print(f"Duplicate assistant answers: {assistant_duplicates}")
    print(f"Duplicate user+answer pairs: {pair_duplicates}")

    # --------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------

    print()
    print("SOURCE DISTRIBUTION")
    print("-" * 90)

    source_counts = Counter(sources)

    if source_counts:
        print_counter(source_counts, len(rows))

    else:
        print("No source values found.")

    return {
        "rows": rows,
        "users": users,
        "assistants": assistants,
        "pairs": pairs,
        "sources": sources,
    }


# ============================================================
# LEAKAGE
# ============================================================

def check_leakage(train, validation):
    train_users = set(train["users"])
    val_users = set(validation["users"])

    train_pairs = set(train["pairs"])
    val_pairs = set(validation["pairs"])

    pair_overlap = train_pairs & val_pairs
    user_overlap = train_users & val_users

    print()
    print("=" * 90)
    print("TRAIN / VALIDATION LEAKAGE")
    print("=" * 90)

    print(
        f"Exact user+assistant overlap : "
        f"{len(pair_overlap)}"
    )

    print(
        f"Exact user overlap           : "
        f"{len(user_overlap)}"
    )


# ============================================================
# SAMPLE DISPLAY
# ============================================================

def show_samples(name, data, count=3):
    print()
    print("=" * 90)
    print(f"SAMPLES — {name.upper()}")
    print("=" * 90)

    rows = data["rows"]

    if not rows:
        print("No samples available.")
        return

    for i, row in enumerate(rows[:count]):
        print()
        print(f"--- SAMPLE {i} ---")

        print("USER:")
        print(get_text(row, "user")[:1500])

        print()
        print("ASSISTANT:")
        print(get_text(row, "assistant")[:2000])

        print()
        print("SOURCE:")
        print(get_text(row, "source"))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("CYBERSECURITY SFT DATASET QUALITY DIAGNOSIS")
    print("=" * 90)

    print()
    print("Loading datasets...")

    train = diagnose_dataset(
        "train",
        TRAIN_PATH
    )

    validation = diagnose_dataset(
        "validation",
        VAL_PATH
    )

    check_leakage(
        train,
        validation
    )

    show_samples(
        "train",
        train,
        count=3
    )

    show_samples(
        "validation",
        validation,
        count=3
    )

    print()
    print("=" * 90)
    print("FINAL DIAGNOSIS")
    print("=" * 90)

    print(
        f"\nTrain examples      : "
        f"{len(train['rows'])}"
    )

    print(
        f"Validation examples : "
        f"{len(validation['rows'])}"
    )

    if len(train["rows"]) == 0:
        print("\nWARNING: Train dataset is empty.")

    if len(validation["rows"]) == 0:
        print("\nWARNING: Validation dataset is empty.")

    if len(train["rows"]) > 0 and len(validation["rows"]) > 0:
        print(
            "\nDataset diagnosis completed successfully."
        )

    print()


if __name__ == "__main__":
    main()