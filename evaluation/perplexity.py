#!/usr/bin/env python3

import argparse
import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--data",
        default="evaluation/data/wikitext-2-test.parquet",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("PERPLEXITY")
    print("=" * 80)

    print(f"Model   : {args.model}")
    print(f"Dataset : {args.data}")
    print(f"Samples : {args.samples}")
    print(f"Length  : {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    dataset = load_dataset(
        "parquet",
        data_files=args.data,
        split="train",
    )

    texts = []

    for row in dataset:

        text = str(
            row.get("text", "")
        ).strip()

        if text:
            texts.append(text)

        if len(texts) >= args.samples:
            break

    total_nll = 0.0
    total_tokens = 0

    device = next(
        model.parameters()
    ).device

    with torch.no_grad():

        for index, text in enumerate(texts):

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=True,
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]

            mask = attention_mask[:, 1:].bool()

            log_probs = torch.nn.functional.log_softmax(
                logits,
                dim=-1,
            )

            token_log_probs = (
                log_probs
                .gather(
                    dim=-1,
                    index=labels.unsqueeze(-1),
                )
                .squeeze(-1)
            )

            token_log_probs = token_log_probs[
                mask
            ]

            total_nll -= token_log_probs.sum().item()

            total_tokens += token_log_probs.numel()

            if (index + 1) % 16 == 0:
                print(
                    f"Processed "
                    f"{index + 1}/{len(texts)}"
                )

    perplexity = math.exp(
        total_nll / total_tokens
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"Tokens      : {total_tokens:,}"
    )

    print(
        f"Perplexity  : {perplexity:.6f}"
    )


if __name__ == "__main__":
    main()