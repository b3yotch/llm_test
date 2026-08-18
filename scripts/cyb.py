#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare base Qwen model against a LoRA adapter."
    )

    parser.add_argument(
        "--base-model",
        required=True,
        help="Path or Hugging Face ID of the base Qwen model.",
    )

    parser.add_argument(
        "--lora-adapter",
        required=True,
        help="Path to the trained LoRA adapter/checkpoint.",
    )

    parser.add_argument(
        "--output",
        default="evaluation/base_vs_lora_comparison.json",
        help="Output JSON file.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def build_messages(prompt: str):
    """
    Keep the system instruction identical for both models.

    This prevents the comparison from being affected by different
    prompting strategies.
    """

    return [
        {
            "role": "system",
            "content": (
                "You are a cybersecurity assistant. "
                "Answer directly and professionally. "
                "Do not expose internal reasoning or narrate your thought process. "
                "Do not begin responses with meta-commentary such as "
                "\"Okay, so I need to\" or \"Let me think\"."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def load_model_and_tokenizer(
    model_path,
    adapter_path=None,
):
    print(f"\nLoading tokenizer from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter from: {adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )

    model.eval()

    return model, tokenizer


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
):
    messages = build_messages(prompt)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    # device_map="auto" places the model on the GPU.
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    input_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
    )

    elapsed = time.perf_counter() - start

    generated_ids = outputs[0][input_tokens:]

    output_tokens = len(generated_ids)

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    tokens_per_second = (
        output_tokens / elapsed
        if elapsed > 0
        else 0.0
    )

    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_time_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
    }


def run_evaluation(
    model_name,
    model,
    tokenizer,
    max_new_tokens,
):
    results = []

    print()
    print("=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)

    for index, item in enumerate(PROMPTS):

        print(
            f"\n[{index + 1}/{len(PROMPTS)}] "
            f"{item['category']}"
        )

        result = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            max_new_tokens=max_new_tokens,
        )

        result.update(
            {
                "index": index,
                "category": item["category"],
                "prompt": item["prompt"],
            }
        )

        results.append(result)

        print(
            f"Output tokens : {result['output_tokens']}"
        )

        print(
            f"Generation    : "
            f"{result['generation_time_seconds']:.3f}s"
        )

        print(
            f"Throughput    : "
            f"{result['tokens_per_second']:.2f} tok/s"
        )

        print(
            f"Preview       : "
            f"{result['text'][:180].replace(chr(10), ' ')}..."
        )

    return results


def calculate_summary(results):
    if not results:
        return {
            "examples": 0,
            "average_output_tokens": 0,
            "average_generation_time_seconds": 0,
            "average_tokens_per_second": 0,
            "truncated_at_max_tokens": 0,
        }

    return {
        "examples": len(results),
        "average_output_tokens": (
            sum(
                x["output_tokens"]
                for x in results
            )
            / len(results)
        ),
        "average_generation_time_seconds": (
            sum(
                x["generation_time_seconds"]
                for x in results
            )
            / len(results)
        ),
        "average_tokens_per_second": (
            sum(
                x["tokens_per_second"]
                for x in results
            )
            / len(results)
        ),
        "truncated_at_max_tokens": sum(
            1
            for x in results
            if x["output_tokens"]
            >= args.max_new_tokens
        ),
    }


def count_meta_style_phrases(text):
    phrases = [
        "okay, so i need to",
        "let me think",
        "let me start by",
        "i need to explain",
        "first, i need to",
        "now, let's",
        "but wait",
        "i should recall",
    ]

    lowered = text.lower()

    return sum(
        1
        for phrase in phrases
        if phrase in lowered
    )


def style_analysis(results):
    total = len(results)

    examples_with_meta_style = sum(
        1
        for result in results
        if count_meta_style_phrases(
            result["text"]
        ) > 0
    )

    return {
        "examples_with_meta_reasoning_style": (
            examples_with_meta_style
        ),
        "meta_reasoning_style_rate": (
            examples_with_meta_style / total
            if total
            else 0
        ),
    }


def main():
    global args
    args = parse_args()

    torch.manual_seed(args.seed)

    print("=" * 80)
    print("BASE vs LoRA CYBERSECURITY COMPARISON")
    print("=" * 80)

    print(
        f"Base model    : {args.base_model}"
    )

    print(
        f"LoRA adapter  : {args.lora_adapter}"
    )

    print(
        f"Max new tokens: {args.max_new_tokens}"
    )

    print(
        f"Sampling      : False"
    )

    print(
        f"Temperature   : {args.temperature}"
    )

    print(
        f"Seed          : {args.seed}"
    )

    # ------------------------------------------------------------------
    # Base model
    # ------------------------------------------------------------------

    base_model, base_tokenizer = (
        load_model_and_tokenizer(
            args.base_model
        )
    )

    base_results = run_evaluation(
        model_name="BASE",
        model=base_model,
        tokenizer=base_tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    base_summary = calculate_summary(
        base_results
    )

    base_style = style_analysis(
        base_results
    )

    # Free base model before loading LoRA model.
    del base_model
    del base_tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # LoRA model
    # ------------------------------------------------------------------

    lora_model, lora_tokenizer = (
        load_model_and_tokenizer(
            args.base_model,
            args.lora_adapter,
        )
    )

    lora_results = run_evaluation(
        model_name="LORA",
        model=lora_model,
        tokenizer=lora_tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    lora_summary = calculate_summary(
        lora_results
    )

    lora_style = style_analysis(
        lora_results
    )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    speed_change = (
        (
            lora_summary["average_tokens_per_second"]
            / base_summary["average_tokens_per_second"]
        )
        - 1
    ) * 100

    output = {
        "configuration": {
            "base_model": args.base_model,
            "lora_adapter": args.lora_adapter,
            "max_new_tokens": args.max_new_tokens,
            "sampling": False,
            "temperature": args.temperature,
            "dtype": "bfloat16",
            "thinking": False,
            "seed": args.seed,
        },

        "base_summary": base_summary,

        "lora_summary": lora_summary,

        "comparison": {
            "throughput_change_percent": speed_change,
            "base_meta_reasoning_style_rate": (
                base_style[
                    "meta_reasoning_style_rate"
                ]
            ),
            "lora_meta_reasoning_style_rate": (
                lora_style[
                    "meta_reasoning_style_rate"
                ]
            ),
        },

        "base_results": base_results,

        "lora_results": lora_results,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    output_path = Path(args.output)
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
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("\nBASE")
    print(
        f"Average tok/s : "
        f"{base_summary['average_tokens_per_second']:.2f}"
    )

    print(
        f"Average tokens: "
        f"{base_summary['average_output_tokens']:.1f}"
    )

    print(
        f"Meta-style    : "
        f"{base_style['meta_reasoning_style_rate'] * 100:.1f}%"
    )

    print("\nLORA")
    print(
        f"Average tok/s : "
        f"{lora_summary['average_tokens_per_second']:.2f}"
    )

    print(
        f"Average tokens: "
        f"{lora_summary['average_output_tokens']:.1f}"
    )

    print(
        f"Meta-style    : "
        f"{lora_style['meta_reasoning_style_rate'] * 100:.1f}%"
    )

    print()
    print(
        f"LoRA throughput change: "
        f"{speed_change:+.2f}%"
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()