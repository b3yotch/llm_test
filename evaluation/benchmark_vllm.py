#!/usr/bin/env python3

import argparse
import json
import statistics
import time

import requests


PROMPTS = [
    "What is PagedAttention in one sentence?",
    "Explain the difference between stateful and stateless firewalls.",
    "What is the principle of least privilege in cloud environments?",
    "Explain how Kerberos authentication works in Active Directory.",
    "What is the difference between symmetric and asymmetric encryption?",
    "What is an SQL injection vulnerability and how can parameterized queries prevent it?",
    "What is the difference between an IOC and a TTP in threat intelligence?",
    "Describe the main phases of incident response.",
    "What is the difference between static and dynamic malware analysis?",
    "Explain the difference between vulnerability scanning and penetration testing.",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)

    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
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
        required=True,
    )

    return parser.parse_args()


def generate(
    base_url,
    model,
    prompt,
    max_tokens,
    temperature,
):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"/no_think\n{prompt}",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()

    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=300,
    )

    elapsed = time.perf_counter() - start

    response.raise_for_status()

    data = response.json()

    message = data["choices"][0]["message"]["content"]

    usage = data.get("usage", {})

    prompt_tokens = usage.get(
        "prompt_tokens",
        0,
    )

    completion_tokens = usage.get(
        "completion_tokens",
        0,
    )

    tokens_per_second = (
        completion_tokens / elapsed
        if elapsed > 0
        else 0.0
    )

    return {
        "prompt": prompt,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens
        ),
        "tokens_per_second": tokens_per_second,
        "response": message,
    }


def main():
    args = parse_args()

    results = []

    print("=" * 80)
    print("vLLM SERVING BENCHMARK")
    print("=" * 80)

    print(f"Model     : {args.model}")
    print(f"Endpoint  : {args.base_url}")
    print(f"Prompts   : {len(PROMPTS)}")
    print(f"Max tokens: {args.max_tokens}")

    for index, prompt in enumerate(PROMPTS):

        print(
            f"\n[{index + 1}/{len(PROMPTS)}] "
            f"{prompt}"
        )

        result = generate(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        results.append(result)

        print(
            f"Latency   : "
            f"{result['elapsed_seconds']:.4f}s"
        )

        print(
            f"Completion: "
            f"{result['completion_tokens']} tokens"
        )

        print(
            f"Throughput: "
            f"{result['tokens_per_second']:.2f} tok/s"
        )

    latencies = [
        result["elapsed_seconds"]
        for result in results
    ]

    throughputs = [
        result["tokens_per_second"]
        for result in results
    ]

    total_completion_tokens = sum(
        result["completion_tokens"]
        for result in results
    )

    total_elapsed = sum(
        result["elapsed_seconds"]
        for result in results
    )

    summary = {
        "model": args.model,
        "num_prompts": len(results),

        "mean_latency_seconds": (
            statistics.mean(latencies)
        ),

        "median_latency_seconds": (
            statistics.median(latencies)
        ),

        "mean_generation_tokens_per_second": (
            statistics.mean(throughputs)
        ),

        "total_completion_tokens": (
            total_completion_tokens
        ),

        "total_elapsed_seconds": (
            total_elapsed
        ),

        "results": results,
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Mean latency : "
        f"{summary['mean_latency_seconds']:.4f}s"
    )

    print(
        f"Median latency: "
        f"{summary['median_latency_seconds']:.4f}s"
    )

    print(
        f"Mean tok/s   : "
        f"{summary['mean_generation_tokens_per_second']:.2f}"
    )

    print(
        f"Total tokens : "
        f"{summary['total_completion_tokens']}"
    )

    print(
        f"Total time   : "
        f"{summary['total_elapsed_seconds']:.4f}s"
    )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved to: {args.output}"
    )


if __name__ == "__main__":
    main()