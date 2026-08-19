#!/usr/bin/env python3

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import requests


# ============================================================================
# Evaluation prompts
# ============================================================================

PROMPTS = [
    {
        "category": "network_security",
        "prompt": (
            "Explain the difference between a stateful and stateless firewall "
            "and when each approach is appropriate."
        ),
    },
    {
        "category": "identity_security",
        "prompt": (
            "Explain how Kerberos authentication works in an Active Directory "
            "environment and why service accounts require careful management."
        ),
    },
    {
        "category": "cryptography",
        "prompt": (
            "Explain the difference between symmetric and asymmetric encryption "
            "and give practical cybersecurity use cases for each."
        ),
    },
    {
        "category": "web_security",
        "prompt": (
            "What is an SQL injection vulnerability? Explain why parameterized "
            "queries prevent it and show a safe Python example."
        ),
    },
    {
        "category": "secure_coding",
        "prompt": (
            "What are the most important practices for securely handling user "
            "input in a Python web application?"
        ),
    },
    {
        "category": "threat_intelligence",
        "prompt": (
            "What is the difference between an IOC and a TTP in threat "
            "intelligence? Give examples of both."
        ),
    },
    {
        "category": "incident_response",
        "prompt": (
            "Describe the main phases of a cybersecurity incident response "
            "process and explain the purpose of each phase."
        ),
    },
    {
        "category": "malware_analysis",
        "prompt": (
            "What is the difference between static and dynamic malware analysis? "
            "Explain the advantages and limitations of both."
        ),
    },
    {
        "category": "offensive_security",
        "prompt": (
            "Explain the difference between vulnerability scanning and "
            "penetration testing, including what each is intended to accomplish."
        ),
    },
    {
        "category": "cloud_security",
        "prompt": (
            "Explain the principle of least privilege in cloud environments "
            "and why overly broad IAM permissions are dangerous."
        ),
    },
    {
        "category": "social_engineering",
        "prompt": (
            "What psychological techniques are commonly exploited in phishing "
            "attacks, and how can organizations train employees to recognize them?"
        ),
    },
]


SYSTEM_PROMPT = (
    "You are a cybersecurity assistant. "
    "Answer directly, accurately, and professionally. "
    "Do not reveal internal reasoning. "
    "Do not narrate your thought process."
)


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare cybersecurity quality across base, BF16 LoRA, and GPTQ models."
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL.",
    )

    parser.add_argument(
        "--original-model",
        required=True,
        help="Original Qwen model ID exposed by vLLM.",
    )

    parser.add_argument(
        "--merged-model",
        required=True,
        help="Merged BF16 cybersecurity model ID exposed by vLLM.",
    )

    parser.add_argument(
        "--gptq-model",
        required=True,
        help="GPTQ W8A16 cybersecurity model ID exposed by vLLM.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--output",
        default="evaluation/cybersecurity_quality_comparison.json",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
    )

    return parser.parse_args()


# ============================================================================
# Helpers
# ============================================================================

def request_model(
    base_url,
    model,
    prompt,
    max_tokens,
    temperature,
    timeout,
):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"/no_think\n{prompt}",
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()

    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=timeout,
    )

    elapsed = time.perf_counter() - start

    response.raise_for_status()

    data = response.json()

    message = data["choices"][0]["message"]
    text = message.get("content") or ""

    usage = data.get("usage", {})

    prompt_tokens = usage.get(
        "prompt_tokens",
        0,
    )

    completion_tokens = usage.get(
        "completion_tokens",
        0,
    )

    finish_reason = data["choices"][0].get(
        "finish_reason"
    )

    return {
        "text": text,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }


def repetition_score(text: str) -> float:
    """
    Rough repetition indicator.

    Counts repeated adjacent phrases of 5+ words.
    Higher means more repetition.
    """

    words = re.findall(
        r"\b[\w'-]+\b",
        text.lower(),
    )

    if len(words) < 20:
        return 0.0

    ngram_size = 5
    counts = {}

    for i in range(
        len(words) - ngram_size + 1
    ):
        ngram = tuple(
            words[i:i + ngram_size]
        )
        counts[ngram] = counts.get(
            ngram,
            0,
        ) + 1

    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated / max(
        1,
        len(words),
    )


def quality_flags(
    text,
    completion_tokens,
    max_tokens,
    finish_reason,
):
    lowered = text.lower()

    return {
        "empty_response": not text.strip(),

        "contains_think_tag": (
            "<think>" in lowered
            or "</think>" in lowered
        ),

        "visible_reasoning_markers": any(
            phrase in lowered
            for phrase in [
                "okay, so i need to",
                "let me think",
                "let me start by",
                "i need to explain",
                "first, i should",
                "wait,",
                "i should make sure",
            ]
        ),

        "truncated_by_token_limit": (
            completion_tokens >= max_tokens
            or finish_reason == "length"
        ),

        "repetition_score": repetition_score(
            text
        ),
    }


def automatic_summary(results):
    if not results:
        return {}

    summary = {
        "examples": len(results),
        "mean_completion_tokens": statistics.mean(
            x["completion_tokens"]
            for x in results
        ),
        "median_completion_tokens": statistics.median(
            x["completion_tokens"]
            for x in results
        ),
        "mean_latency_seconds": statistics.mean(
            x["elapsed_seconds"]
            for x in results
        ),
        "empty_responses": sum(
            x["quality"]["empty_response"]
            for x in results
        ),
        "think_tag_responses": sum(
            x["quality"]["contains_think_tag"]
            for x in results
        ),
        "visible_reasoning_responses": sum(
            x["quality"]["visible_reasoning_markers"]
            for x in results
        ),
        "truncated_responses": sum(
            x["quality"]["truncated_by_token_limit"]
            for x in results
        ),
        "mean_repetition_score": statistics.mean(
            x["quality"]["repetition_score"]
            for x in results
        ),
    }

    return summary


# ============================================================================
# Evaluate one model
# ============================================================================

def evaluate_model(
    base_url,
    model,
    label,
    max_tokens,
    temperature,
    timeout,
):
    print()
    print("=" * 80)
    print(f"EVALUATING: {label}")
    print(f"MODEL: {model}")
    print("=" * 80)

    results = []

    for index, item in enumerate(PROMPTS):

        print(
            f"\n[{index + 1}/{len(PROMPTS)}] "
            f"{item['category']}"
        )

        try:
            output = request_model(
                base_url=base_url,
                model=model,
                prompt=item["prompt"],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )

            result = {
                "index": index,
                "category": item["category"],
                "prompt": item["prompt"],
                "response": output["text"],
                "elapsed_seconds": output[
                    "elapsed_seconds"
                ],
                "prompt_tokens": output[
                    "prompt_tokens"
                ],
                "completion_tokens": output[
                    "completion_tokens"
                ],
                "finish_reason": output[
                    "finish_reason"
                ],
                "quality": quality_flags(
                    text=output["text"],
                    completion_tokens=output[
                        "completion_tokens"
                    ],
                    max_tokens=max_tokens,
                    finish_reason=output[
                        "finish_reason"
                    ],
                ),
            }

            results.append(result)

            print(
                f"Tokens : {result['completion_tokens']}"
            )

            print(
                f"Time   : "
                f"{result['elapsed_seconds']:.3f}s"
            )

            print(
                f"Flags  : "
                f"think={result['quality']['contains_think_tag']}, "
                f"reasoning={result['quality']['visible_reasoning_markers']}, "
                f"truncated={result['quality']['truncated_by_token_limit']}"
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

            results.append(
                {
                    "index": index,
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "error": str(exc),
                }
            )

    return {
        "model": model,
        "label": label,
        "summary": automatic_summary(
            [
                result
                for result in results
                if "error" not in result
            ]
        ),
        "results": results,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    original = evaluate_model(
        base_url=args.base_url,
        model=args.original_model,
        label="ORIGINAL QWEN3-4B",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    merged = evaluate_model(
        base_url=args.base_url,
        model=args.merged_model,
        label="MERGED CYBERSECURITY BF16",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    gptq = evaluate_model(
        base_url=args.base_url,
        model=args.gptq_model,
        label="GPTQ W8A16 CYBERSECURITY",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    output = {
        "configuration": {
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "system_prompt": SYSTEM_PROMPT,
            "no_think": True,
            "num_prompts": len(PROMPTS),
        },

        "evaluation_order": [
            "original",
            "merged_bf16",
            "gptq_w8a16",
        ],

        "models": {
            "original": original,
            "merged_bf16": merged,
            "gptq_w8a16": gptq,
        },
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("QUALITY EVALUATION SUMMARY")
    print("=" * 80)

    for name, result in [
        ("Original", original),
        ("Merged BF16", merged),
        ("GPTQ W8A16", gptq),
    ]:

        s = result["summary"]

        print()
        print(name)

        print(
            f"  Mean completion tokens : "
            f"{s.get('mean_completion_tokens', 0):.1f}"
        )

        print(
            f"  Mean latency           : "
            f"{s.get('mean_latency_seconds', 0):.3f}s"
        )

        print(
            f"  Empty responses        : "
            f"{s.get('empty_responses', 0)}"
        )

        print(
            f"  Think-tag responses    : "
            f"{s.get('think_tag_responses', 0)}"
        )

        print(
            f"  Reasoning-style        : "
            f"{s.get('visible_reasoning_responses', 0)}"
        )

        print(
            f"  Truncated responses    : "
            f"{s.get('truncated_responses', 0)}"
        )

        print(
            f"  Mean repetition score  : "
            f"{s.get('mean_repetition_score', 0):.4f}"
        )

    print()
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()