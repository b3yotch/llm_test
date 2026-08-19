import torch

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

MODEL_PATH = (
    "/mnt/NewDisk/llm_quant_experiment/models/"
    "qwen3-4b-cybersecurity-lora-r8-merged"
)

OUTPUT_DIR = (
    "/mnt/NewDisk/llm_quant_experiment/models/gptq/"
    "qwen3-4b-cybersecurity-lora-r8-w8a16-gptq"
)


# ------------------------------------------------------------------
# GPTQ W8A16 configuration
# ------------------------------------------------------------------
#
# In llmcompressor 0.12.0.1, W8A16 is configured explicitly through
# config_groups rather than using scheme="W8A16_ASYM".
#
# 8-bit integer weights
# 16-bit activations (activation quantization disabled)
# symmetric weight quantization
# group-wise quantization with group size 128
#
# ------------------------------------------------------------------

recipe = [
    GPTQModifier(
        config_groups={
            "group_0": {
                "targets": ["Linear"],

                "input_activations": None,
                "output_activations": None,

                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                },
            }
        },

        ignore=["lm_head"],

        block_size=128,
        dampening_frac=0.01,
        actorder="static",

        offload_hessians=False,
    )
]


# ------------------------------------------------------------------
# Environment information
# ------------------------------------------------------------------

print("=" * 70)
print("Qwen3-4B Cybersecurity GPTQ W8A16 Quantization")
print("=" * 70)

print("Input model:")
print(MODEL_PATH)

print("\nOutput directory:")
print(OUTPUT_DIR)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for GPTQ quantization."
    )

print("\nGPU:")
print(torch.cuda.get_device_name(0))

print(
    "GPU memory:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2,
    ),
    "GB",
)


# ------------------------------------------------------------------
# Quantization
# ------------------------------------------------------------------

print("\nStarting LLM Compressor oneshot...")

print("Calibration dataset : WikiText-2")
print("Calibration config  : wikitext-2-raw-v1")
print("Calibration samples : 128")
print("Sequence length     : 1024")
print("Quantization        : GPTQ W8A16")
print("Weights             : INT8")
print("Activations         : BF16/FP16 path (unquantized)")
print("Weight strategy     : group")
print("Group size          : 128")


oneshot(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,

    recipe=recipe,

    # --------------------------------------------------------------
    # Calibration dataset
    # --------------------------------------------------------------

    dataset="wikitext",
    dataset_config_name="wikitext-2-raw-v1",

    num_calibration_samples=128,
    max_seq_length=1024,

    text_column="text",

    batch_size=1,

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------

    output_dir=OUTPUT_DIR,

    save_compressed=True,

    # --------------------------------------------------------------
    # Memory-conscious sequential quantization
    # --------------------------------------------------------------

    sequential_targets=["Linear"],
)


print("\n" + "=" * 70)
print("GPTQ W8A16 quantization completed")
print("=" * 70)

print("Saved to:")
print(OUTPUT_DIR)