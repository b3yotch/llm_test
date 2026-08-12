import argparse
import math
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate(
    model_path,
    dataset_path,
    max_samples=128,
    max_length=1024,
):

    print("=" * 70)
    print("PERPLEXITY EVALUATION")
    print("=" * 70)

    print("Model:", model_path)
    print("Dataset:", dataset_path)
    print("Samples:", max_samples)
    print("Sequence length:", max_length)

    print("=" * 70)

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print("\nLoading model...")

    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()

    load_time = time.perf_counter() - start

    print(
        f"Model loaded in {load_time:.2f}s"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_parquet(dataset_path)

    texts = [
        text
        for text in df["text"].tolist()
        if isinstance(text, str) and text.strip()
    ]

    texts = texts[:max_samples]

    print(
        f"Loaded {len(texts)} non-empty documents"
    )

    # ---------------------------------------------------------
    # Perplexity
    # ---------------------------------------------------------

    total_nll = 0.0
    total_tokens = 0

    print("\nEvaluating...\n")

    with torch.no_grad():

        for i, text in enumerate(texts):

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            input_ids = encoded["input_ids"]

            if input_ids.shape[1] < 2:
                continue

            input_ids = input_ids.to(
                model.device
            )

            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
            )

            n_tokens = input_ids.shape[1] - 1

            nll = outputs.loss.item()

            total_nll += nll * n_tokens
            total_tokens += n_tokens

            if (i + 1) % 16 == 0:

                current_ppl = math.exp(
                    total_nll / total_tokens
                )

                print(
                    f"{i + 1:3d}/{len(texts)} | "
                    f"tokens={total_tokens:,} | "
                    f"PPL={current_ppl:.4f}"
                )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    mean_nll = total_nll / total_tokens

    perplexity = math.exp(mean_nll)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Total tokens: {total_tokens:,}"
    )

    print(
        f"Mean NLL:     {mean_nll:.6f}"
    )

    print(
        f"Perplexity:   {perplexity:.6f}"
    )

    print("=" * 70)

    return perplexity


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--dataset",
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

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    ppl = evaluate(
        model_path=args.model,
        dataset_path=args.dataset,
        max_samples=args.samples,
        max_length=args.max_length,
    )

    with open(args.output, "w") as f:

        f.write(
            f"model={args.model}\n"
            f"dataset={args.dataset}\n"
            f"samples={args.samples}\n"
            f"max_length={args.max_length}\n"
            f"perplexity={ppl:.6f}\n"
        )

    print(
        f"\nSaved result to {args.output}"
    )