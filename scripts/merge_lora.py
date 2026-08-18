#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model."
    )

    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to the original/base Qwen model.",
    )

    parser.add_argument(
        "--lora-adapter",
        required=True,
        help="Path to the trained LoRA checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the merged model.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MERGING QWEN LoRA ADAPTER")
    print("=" * 80)

    print(f"Base model   : {args.base_model}")
    print(f"LoRA adapter : {args.lora_adapter}")
    print(f"Output       : {args.output_dir}")

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # Load base model
    # ------------------------------------------------------------------

    print("\nLoading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------
    # Load LoRA adapter
    # ------------------------------------------------------------------

    print("\nLoading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        args.lora_adapter,
    )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    print("\nMerging LoRA weights into base model...")

    merged_model = model.merge_and_unload()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    print("\nSaving merged model...")

    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        output_dir
    )

    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)

    print(
        f"Merged model saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()