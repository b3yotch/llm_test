#!/usr/bin/env python3

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# Configuration
# ============================================================

TEST_PROMPTS = [
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
            "Explain the difference between vulnerability scanning and penetration "
            "testing, including what each is intended to accomplish."
        ),
    },
    {
        "category": "cloud_security",
        "prompt": (
            "Explain the principle of least privilege in cloud environments and "
            "why overly broad IAM permissions are dangerous."
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


# ============================================================
# Utilities
# ============================================================

def cleanup_model(model=None, tokenizer=None):
    """
    Completely release model/tokenizer references and clear CUDA memory.
    """

    if model is not None:
        del model

    if tokenizer is not None:
        del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Small diagnostic
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        print(
            f"CUDA memory after cleanup: "
            f"allocated={allocated:.2f} GiB, "
            f"reserved={reserved:.2f} GiB"
        )


def get_model_device(model):
    """
    Get the device used by the model.
    """

    return next(model.parameters()).device


# ============================================================
# Model loading
# ============================================================

def load_base_model(model_path):
    print("=" * 80)
    print("LOADING BASE MODEL")
    print("=" * 80)
    print(f"Path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    print(f"Model device: {get_model_device(model)}")

    return tokenizer, model


def load_lora_model(model_path, adapter_path):
    print("=" * 80)
    print("LOADING BASE MODEL + LORA")
    print("=" * 80)
    print(f"Base model : {model_path}")
    print(f"LoRA       : {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    model.eval()

    print(f"Model device: {get_model_device(model)}")

    return tokenizer, model


# ============================================================
# Prompt formatting
# ============================================================

def format_prompt(tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # Qwen3 supports disabling thinking through the chat template.
    # The fallback makes the script work with tokenizers that do not
    # expose the enable_thinking argument.
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return formatted


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=256,
):
    formatted_prompt = format_prompt(
        tokenizer,
        prompt,
    )

    device = get_model_device(model)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()

    outputs = model.generate(
        **inputs,

        # Deterministic generation.
        # Important for base-vs-LoRA comparison.
        do_sample=False,

        max_new_tokens=max_new_tokens,

        pad_token_id=tokenizer.eos_token_id,
    )

    elapsed = time.perf_counter() - start

    output_tokens = outputs.shape[1] - input_tokens

    generated_text = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True,
    )

    tokens_per_second = (
        output_tokens / elapsed
        if elapsed > 0
        else 0.0
    )

    return {
        "text": generated_text.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_time_seconds": round(elapsed, 4),
        "tokens_per_second": round(tokens_per_second, 2),
    }


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    model_name,
    max_new_tokens,
):
    print("\n")
    print("=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)

    results = []

    for index, item in enumerate(TEST_PROMPTS):

        category = item["category"]
        prompt = item["prompt"]

        print("\n")
        print("-" * 80)
        print(
            f"TEST {index + 1}/{len(TEST_PROMPTS)}"
            f" | {category}"
        )
        print("-" * 80)

        print("\nPROMPT:")
        print(prompt)

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        print("\nRESPONSE:")
        print(result["text"])

        print(
            f"\nGeneration time : "
            f"{result['generation_time_seconds']:.2f}s"
        )

        print(
            f"Output tokens   : "
            f"{result['output_tokens']}"
        )

        print(
            f"Throughput      : "
            f"{result['tokens_per_second']:.2f} tok/s"
        )

        results.append(
            {
                "index": index,
                "category": category,
                "prompt": prompt,
                **result,
            }
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Sequential base vs LoRA evaluation"
    )

    parser.add_argument(
        "--model",
        default="models/qwen3-4b-original",
        help="Path to the original Qwen3-4B model",
    )

    parser.add_argument(
        "--adapter",
        default="models/qwen3-4b-lora-r8/checkpoint-3469",
        help="Path to the final LoRA adapter",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output",
        default="evaluation/lora_test_results.json",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SEQUENTIAL BASE vs LORA TEST")
    print("=" * 80)

    print(f"Base model : {args.model}")
    print(f"LoRA       : {args.adapter}")
    print(f"Prompts    : {len(TEST_PROMPTS)}")
    print(f"Max tokens : {args.max_new_tokens}")

    # ========================================================
    # PHASE 1 — BASE MODEL
    # ========================================================

    base_tokenizer, base_model = load_base_model(
        args.model
    )

    base_results = evaluate_model(
        model=base_model,
        tokenizer=base_tokenizer,
        model_name="BASE QWEN3-4B",
        max_new_tokens=args.max_new_tokens,
    )

    # Save immediately in case anything goes wrong later.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.model,
                "adapter": args.adapter,
                "base_results": base_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nBase evaluation saved.")

    # ========================================================
    # UNLOAD BASE MODEL
    # ========================================================

    print("\n")
    print("=" * 80)
    print("UNLOADING BASE MODEL")
    print("=" * 80)

    cleanup_model(
        model=base_model,
        tokenizer=base_tokenizer,
    )

    # ========================================================
    # PHASE 2 — LORA MODEL
    # ========================================================

    lora_tokenizer, lora_model = load_lora_model(
        args.model,
        args.adapter,
    )

    lora_results = evaluate_model(
        model=lora_model,
        tokenizer=lora_tokenizer,
        model_name="QWEN3-4B + LORA",
        max_new_tokens=args.max_new_tokens,
    )

    # ========================================================
    # Final combined results
    # ========================================================

    final_results = {
        "configuration": {
            "base_model": args.model,
            "lora_adapter": args.adapter,
            "max_new_tokens": args.max_new_tokens,
            "sampling": False,
            "dtype": "bfloat16",
            "sequential_loading": True,
            "thinking": False,
        },
        "base_results": base_results,
        "lora_results": lora_results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n")
    print("=" * 80)
    print("BASE vs LORA SUMMARY")
    print("=" * 80)

    base_total_tokens = sum(
        x["output_tokens"]
        for x in base_results
    )

    lora_total_tokens = sum(
        x["output_tokens"]
        for x in lora_results
    )

    base_total_time = sum(
        x["generation_time_seconds"]
        for x in base_results
    )

    lora_total_time = sum(
        x["generation_time_seconds"]
        for x in lora_results
    )

    print(
        f"Base total generation time : "
        f"{base_total_time:.2f}s"
    )

    print(
        f"LoRA total generation time : "
        f"{lora_total_time:.2f}s"
    )

    print(
        f"Base total output tokens   : "
        f"{base_total_tokens}"
    )

    print(
        f"LoRA total output tokens   : "
        f"{lora_total_tokens}"
    )

    print("\nResults:")
    print(output_path)

    # ========================================================
    # Cleanup LoRA model
    # ========================================================

    cleanup_model(
        model=lora_model,
        tokenizer=lora_tokenizer,
    )

    print("\n")
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()