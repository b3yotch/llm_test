from datasets import load_dataset
from collections import Counter
import json
import os
import re


DATASET = "ChaoticNeutrals/Cybersecurity-ShareGPT"

REPORT_DIR = "reports"
OUTPUT_JSON = os.path.join(
    REPORT_DIR,
    "sharegpt_categorization.json",
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize(text):
    if text is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def get_human_question(conversations):
    """
    Extract the human/user question from ShareGPT conversation format.
    """

    if not isinstance(conversations, list):
        return ""

    for message in conversations:

        if not isinstance(message, dict):
            continue

        role = message.get("from", "").lower()

        if role in ("human", "user"):
            return normalize(
                message.get("value", "")
            )

    return ""


def get_assistant_answer(conversations):

    if not isinstance(conversations, list):
        return ""

    for message in conversations:

        if not isinstance(message, dict):
            continue

        role = message.get("from", "").lower()

        if role in ("gpt", "assistant"):
            return normalize(
                message.get("value", "")
            )

    return ""


# ---------------------------------------------------------------------
# Code detection
# ---------------------------------------------------------------------

CODE_PATTERNS = [
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
    r"#!/usr/bin/",
    r"\bpowershell\b",
    r"\bcmd\.exe\b",
]


def contains_code(text):

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in CODE_PATTERNS
    )


# ---------------------------------------------------------------------
# Generic programming
# ---------------------------------------------------------------------

GENERIC_PROGRAMMING = [
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
    "data structure",
    "algorithm implementation",
]


# ---------------------------------------------------------------------
# Cybersecurity categories
# ---------------------------------------------------------------------

CATEGORY_PATTERNS = {

    "web_security": [
        "sql injection",
        "xss",
        "cross-site scripting",
        "csrf",
        "ssrf",
        "command injection",
        "path traversal",
        "directory traversal",
        "web application security",
        "web security",
    ],

    "network_security": [
        "network security",
        "firewall",
        "ids",
        "ips",
        "packet",
        "tcp",
        "udp",
        "dns security",
        "tls",
        "ssl",
        "network traffic",
        "traffic analysis",
    ],

    "cloud_security": [
        "aws security",
        "azure security",
        "gcp security",
        "cloud security",
        "iam",
        "s3 bucket",
        "kubernetes security",
        "container security",
    ],

    "identity_security": [
        "active directory",
        "active directory",
        "ldap",
        "kerberos",
        "ntlm",
        "domain controller",
        "privilege escalation",
        "group policy",
        "identity security",
    ],

    "malware_analysis": [
        "malware",
        "ransomware",
        "trojan",
        "rootkit",
        "reverse engineering",
        "malware analysis",
        "static analysis",
        "dynamic analysis",
    ],

    "incident_response": [
        "incident response",
        "security incident",
        "forensic",
        "digital forensics",
        "containment",
        "eradication",
        "incident investigation",
    ],

    "threat_intelligence": [
        "threat intelligence",
        "threat actor",
        "ioc",
        "indicators of compromise",
        "mitre att&ck",
        "ttp",
        "campaign",
        "threat hunting",
    ],

    "secure_coding": [
        "secure coding",
        "secure code",
        "code review",
        "vulnerability",
        "security vulnerability",
        "input validation",
        "sanitization",
        "authentication vulnerability",
        "authorization vulnerability",
    ],

    "cryptography": [
        "cryptography",
        "encryption",
        "decryption",
        "aes",
        "rsa",
        "hashing",
        "sha256",
        "digital signature",
        "certificate",
        "public key",
        "private key",
    ],

    "social_engineering": [
        "phishing",
        "spear phishing",
        "social engineering",
        "credential harvesting",
        "social engineering attack",
    ],

    "offensive_security": [
        "penetration testing",
        "red team",
        "exploit",
        "exploitation",
        "payload",
        "shellcode",
        "buffer overflow",
        "return-to-libc",
        "rop",
        "reverse shell",
        "command and control",
        "c2",
        "lateral movement",
        "pass-the-hash",
        "credential dumping",
        "persistence",
        "bypass",
        "evasion",
    ],

    "security_general": [
        "cybersecurity",
        "information security",
        "infosec",
        "security",
        "cyber attack",
        "cyber defense",
        "defensive security",
    ],
}


# ---------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------

HIGH_RISK_PATTERNS = [
    "credential dumping",
    "credential theft",
    "steal credentials",
    "password dumping",
    "pass-the-hash",
    "pass the hash",
    "bypass edr",
    "bypass antivirus",
    "evade detection",
    "avoid detection",
    "disable security",
    "disable antivirus",
    "disable defender",
    "persistence mechanism",
    "reverse shell",
    "ransomware",
    "keylogger",
    "phishing kit",
    "spear phishing tool",
    "credential harvesting",
]


MEDIUM_RISK_PATTERNS = [
    "exploit",
    "exploitation",
    "payload",
    "shellcode",
    "buffer overflow",
    "return-to-libc",
    "rop",
    "privilege escalation",
    "lateral movement",
    "command and control",
    "c2",
    "red team",
    "penetration testing",
    "phishing",
]


def matches(text, patterns):

    text = text.lower()

    return [
        pattern
        for pattern in patterns
        if pattern in text
    ]


# ---------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------

def categorize(question, answer):

    combined = (
        question
        + "\n"
        + answer
    ).lower()

    categories = []

    for category, patterns in CATEGORY_PATTERNS.items():

        if any(
            pattern in combined
            for pattern in patterns
        ):
            categories.append(category)

    generic = any(
        pattern in question.lower()
        for pattern in GENERIC_PROGRAMMING
    )

    high_risk_matches = matches(
        combined,
        HIGH_RISK_PATTERNS,
    )

    medium_risk_matches = matches(
        combined,
        MEDIUM_RISK_PATTERNS,
    )

    if high_risk_matches:

        risk = "high"

    elif medium_risk_matches:

        risk = "medium"

    else:

        risk = "low"

    if not categories:

        category = "uncategorized"

    elif generic and len(categories) == 1:

        category = "generic_programming"

    else:

        category = categories[0]

    return {
        "category": category,
        "categories": categories,
        "risk": risk,
        "generic_programming": generic,
        "high_risk_matches": high_risk_matches,
        "medium_risk_matches": medium_risk_matches,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

print("=" * 100)
print("LOADING CYBERSECURITY-SHAREGPT")
print("=" * 100)

dataset = load_dataset(
    DATASET,
    split="train",
)

print("Rows:", len(dataset))


results = []

category_counts = Counter()
risk_counts = Counter()

code_count = 0
generic_count = 0


for index, row in enumerate(dataset):

    conversations = row["conversations"]

    question = get_human_question(
        conversations
    )

    answer = get_assistant_answer(
        conversations
    )

    combined = (
        question
        + "\n"
        + answer
    )

    code = contains_code(
        combined
    )

    if code:
        code_count += 1

    classification = categorize(
        question,
        answer,
    )

    if classification["generic_programming"]:
        generic_count += 1

    category_counts[
        classification["category"]
    ] += 1

    risk_counts[
        classification["risk"]
    ] += 1

    results.append({
        "index": index,
        "question": question,
        "answer_preview": answer[:500],
        "contains_code": code,
        **classification,
    })


# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------

print("\n" + "=" * 100)
print("CATEGORY DISTRIBUTION")
print("=" * 100)

for category, count in category_counts.most_common():

    percentage = (
        count / len(dataset)
    ) * 100

    print(
        f"{category:30s}"
        f"{count:6d}"
        f" ({percentage:6.2f}%)"
    )


print("\n" + "=" * 100)
print("RISK DISTRIBUTION")
print("=" * 100)

for risk, count in risk_counts.most_common():

    percentage = (
        count / len(dataset)
    ) * 100

    print(
        f"{risk:10s}"
        f"{count:6d}"
        f" ({percentage:6.2f}%)"
    )


print("\n" + "=" * 100)
print("OTHER SIGNALS")
print("=" * 100)

print(
    f"Contains code           : "
    f"{code_count} / {len(dataset)}"
)

print(
    f"Generic programming     : "
    f"{generic_count} / {len(dataset)}"
)


# ---------------------------------------------------------------------
# High-risk examples
# ---------------------------------------------------------------------

high_risk = [
    result
    for result in results
    if result["risk"] == "high"
]


print("\n" + "=" * 100)
print("HIGH-RISK EXAMPLES")
print("=" * 100)

for result in high_risk[:20]:

    print(
        f"\n[{result['index']}] "
        f"{result['category']}"
    )

    print(
        "Risk matches:",
        result["high_risk_matches"],
    )

    print(
        "Question:",
        result["question"][:700],
    )


# ---------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "dataset": DATASET,
            "rows": len(dataset),
            "category_counts": dict(
                category_counts
            ),
            "risk_counts": dict(
                risk_counts
            ),
            "code_count": code_count,
            "generic_programming_count": generic_count,
            "results": results,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print(
    "Report saved to:",
    OUTPUT_JSON,
)