import argparse
import json
import time
from statistics import mean, median

from openai import OpenAI


PROMPTS = [
    "What is PagedAttention in one sentence?",
    "Explain what quantization means for a large language model in two sentences.",
    "What is the difference between CPU and GPU memory?",
    "Explain attention in a transformer briefly.",
    "What is AWQ and why is it useful for LLM inference?",
    "Explain what KV cache is in one short paragraph.",
    "What is the purpose of batching in LLM inference?",
    "Explain the difference between FP16 and INT4.",
    "What does vLLM do?",
    "Why can quantization improve LLM serving efficiency?",
]


def benchmark(model_name, base_url, max_tokens):
    client = OpenAI(
        base_url=base_url,
        api_key="dummy",
    )

    results = []

    # Warm-up request
    print("Running warm-up request...")
    client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Say hello."}
        ],
        max_tokens=10,
        temperature=0.0,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
    )

    print("Warm-up complete.\n")

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {prompt}")

        start = time.perf_counter()

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )

        elapsed = time.perf_counter() - start

        usage = response.usage

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        tokens_per_second = (
            completion_tokens / elapsed
            if elapsed > 0
            else 0
        )

        result = {
            "prompt": prompt,
            "elapsed_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "response": response.choices[0].message.content,
        }

        results.append(result)

        print(f"  Time: {elapsed:.3f}s")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  Tokens/sec: {tokens_per_second:.2f}")
        print()

    latencies = [r["elapsed_seconds"] for r in results]
    speeds = [r["tokens_per_second"] for r in results]

    summary = {
        "model": model_name,
        "num_prompts": len(results),
        "mean_latency_seconds": mean(latencies),
        "median_latency_seconds": median(latencies),
        "mean_generation_tokens_per_second": mean(speeds),
        "total_completion_tokens": sum(
            r["completion_tokens"] for r in results
        ),
        "total_elapsed_seconds": sum(latencies),
        "results": results,
    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    summary = benchmark(
        model_name=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
    )

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Model: {summary['model']}")
    print(f"Prompts: {summary['num_prompts']}")
    print(
        f"Mean latency: "
        f"{summary['mean_latency_seconds']:.3f}s"
    )
    print(
        f"Median latency: "
        f"{summary['median_latency_seconds']:.3f}s"
    )
    print(
        f"Mean generation speed: "
        f"{summary['mean_generation_tokens_per_second']:.2f} tok/s"
    )
    print(
        f"Total completion tokens: "
        f"{summary['total_completion_tokens']}"
    )
    print(
        f"Total elapsed: "
        f"{summary['total_elapsed_seconds']:.3f}s"
    )
    print(f"Saved: {args.output}")
