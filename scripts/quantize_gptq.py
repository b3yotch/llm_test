import torch

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


MODEL_PATH = "/mnt/NewDisk/llm_quant_experiment/models/qwen3-4b-original"

OUTPUT_DIR = "/mnt/NewDisk/llm_quant_experiment/models/gptq/qwen3-4b-gptq"


recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W4A16_ASYM",

        block_size=128,
        dampening_frac=0.01,
        actorder="static",

        offload_hessians=False,
    )
]


print("=" * 70)
print("Qwen3-4B GPTQ W4A16_ASYM Quantization")
print("=" * 70)

print("Model:", MODEL_PATH)
print("Output:", OUTPUT_DIR)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "GPU memory:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2
    ),
    "GB"
)

print("\nStarting LLM Compressor oneshot...")

oneshot(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,

    recipe=recipe,

    dataset="wikitext",
    dataset_config_name="wikitext-2-raw-v1",

    num_calibration_samples=128,
    max_seq_length=1024,

    text_column="text",

    batch_size=1,

    output_dir=OUTPUT_DIR,

    save_compressed=True,

    sequential_targets=["Linear"],
)


print("\n" + "=" * 70)
print("GPTQ quantization completed")
print("=" * 70)

print("Saved to:", OUTPUT_DIR)