#!/usr/bin/env python3

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path or Hugging Face ID of the base Qwen model",
    )

    parser.add_argument(
        "--train",
        default="data/final/train.jsonl",
    )

    parser.add_argument(
        "--validation",
        default="data/final/validation.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        default="models/qwen3-4b-lora-r8",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only a small number of training steps",
    )

    return parser.parse_args()


def format_example(example):
    system = example.get("system", "").strip()
    user = example.get("user", "").strip()
    assistant = example.get("assistant", "").strip()

    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system,
        })

    messages.append({
        "role": "user",
        "content": user,
    })

    messages.append({
        "role": "assistant",
        "content": assistant,
    })

    return messages


def main():
    args = parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("QWEN3-4B LoRA TRAINING")
    print("=" * 80)

    print(f"Model              : {args.model}")
    print(f"Train dataset      : {args.train}")
    print(f"Validation dataset : {args.validation}")
    print(f"Output             : {args.output_dir}")
    print(f"LoRA rank          : 8")
    print(f"LoRA alpha         : 16")
    print(f"LoRA dropout       : 0.05")
    print(f"Max length         : {args.max_length}")
    print(f"Seed               : {args.seed}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    print("\nLoading datasets...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train,
            "validation": args.validation,
        },
    )

    print(f"Train rows      : {len(dataset['train']):,}")
    print(f"Validation rows : {len(dataset['validation']):,}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Chat formatting
    # ------------------------------------------------------------------

    print("Applying Qwen chat template...")

    def tokenize(example):
        messages = format_example(example)

        # Full conversation
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Conversation up to the assistant response
        prompt_messages = messages[:-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        prompt_length = min(
            len(prompt_tokens["input_ids"]),
            len(input_ids),
        )

        labels = [-100] * prompt_length + input_ids[prompt_length:]

        # Make absolutely sure labels have the same length as input_ids
        labels = labels[:len(input_ids)]

        if not any(label != -100 for label in labels):
            raise ValueError(
                "No assistant tokens remain after truncation. "
                "Increase --max-length."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    print("\nExample tokenized length:")
    print(len(tokenized["train"][0]["input_ids"]))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("\nLoading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    print("\nConfiguring LoRA...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    training_kwargs = dict(
        output_dir=args.output_dir,

        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=args.gradient_accumulation,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        logging_steps=10,

        eval_strategy="steps",
        eval_steps=1000,

        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,

        bf16=True,

        gradient_checkpointing=True,

        weight_decay=0.01,

        report_to="none",

        seed=args.seed,

        remove_unused_columns=False,
    )

    if args.smoke_test:
        training_kwargs["max_steps"] = 20
        training_kwargs["eval_steps"] = 10
        training_kwargs["save_steps"] = 10

    training_args = TrainingArguments(
        **training_kwargs
    )

    # ------------------------------------------------------------------
    # Collator
    # ------------------------------------------------------------------

    data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt",
)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer.train()

    # ------------------------------------------------------------------
    # Save adapter
    # ------------------------------------------------------------------

    print("\nSaving LoRA adapter...")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(f"LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()