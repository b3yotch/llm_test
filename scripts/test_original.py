import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/NewDisk/llm_quant_experiment/models/original"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded.")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "Allocated VRAM:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)

prompt = "Explain PagedAttention in one paragraph."

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.device)

torch.cuda.synchronize()
start = time.perf_counter()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

new_tokens = output.shape[1] - inputs["input_ids"].shape[1]

print("\n--- Result ---")
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("\nInput tokens :", inputs["input_ids"].shape[1])
print("Output tokens:", new_tokens)
print("Time         :", round(elapsed, 3), "sec")
print(
    "Output tok/s :",
    round(new_tokens / elapsed, 2),
)
print(
    "Peak VRAM    :",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB",
)