# LLM Quantization & Serving Experiment

## 1. Project Overview

This project is an end-to-end experiment for evaluating, fine-tuning, compressing, and serving local Large Language Models (LLMs), with a primary focus on Qwen3-4B.

The project covers:

- Local model execution with Ollama.
- Migration from Ollama to vLLM for production-style serving.
- Establishing an original BF16 model baseline.
- Cybersecurity dataset curation and supervised fine-tuning with LoRA.
- Merging the LoRA adapter into the base model for standalone inference.
- Quantizing the fine-tuned model using LLM Compressor.
- Comparing BF16, LoRA, merged, and quantized models on quality, memory, latency, and throughput.
- Serving the final candidates through vLLM on an NVIDIA RTX 4090.

The main model is **Qwen3-4B**.

**Current phase:** the cybersecurity LoRA fine-tuning stage has been completed for one epoch, the LoRA adapter has been successfully merged into the base BF16 model, and the merged model has demonstrated approximately the same raw token throughput as the original BF16 model in the current Transformers-based benchmark. The next major step is to quantize the merged cybersecurity model, starting with **W8A16/GPTQ**, and then evaluate the quantized models under vLLM.

---

## 2. Current Environment

### Hardware

- GPU: **NVIDIA GeForce RTX 4090**
- GPU capacity reported by PyTorch/vLLM: approximately **23.52 GiB**
- CUDA compute capability: **8.9 / SM89**

Verified with:

```bash
python -c "import torch; print(torch.cuda.get_device_capability(0))"
```

Output:

```text
(8, 9)
```

### Software

```text
Python: 3.12
PyTorch: 2.11.0+cu130
PyTorch CUDA runtime: 13.0
vLLM: 0.26.0
FlashInfer: 0.6.14
```

Project virtual environment:

```text
/mnt/NewDisk/llm_quant_experiment/.venv
```

Project directory:

```text
/mnt/NewDisk/llm_quant_experiment
```

System `nvcc`:

```text
CUDA 12.0
V12.0.140
/usr/bin/nvcc
```

This differs from the CUDA 13.0 runtime bundled with the installed PyTorch build and should be remembered when troubleshooting CUDA extensions or compiler-related issues.

---

## 3. Ollama → vLLM Migration

Ollama was used for earlier local model execution and experimentation.

The project is now moving toward **vLLM for production-style serving**.

### Ollama

Useful for:

- Easy local model execution.
- Rapid experimentation.
- Simple model management.

### vLLM

vLLM is the target serving engine because it provides:

- OpenAI-compatible APIs.
- Efficient GPU inference.
- KV-cache management.
- CUDA graph execution.
- Optimized batching/serving behavior.
- A production-oriented inference server.

Current application flow:

```text
Application
    |
    | OpenAI-compatible HTTP API
    v
  vLLM
    |
    v
 Qwen3-4B
```

---

## 4. Model Lifecycle

The project now contains multiple model stages:

```text
models/qwen3-4b-original
        |
        | LoRA/SFT on cybersecurity dataset
        v
models/qwen3-4b-lora-r8/checkpoint-3469
        |
        | merge_and_unload()
        v
models/qwen3-4b-cybersecurity-lora-r8-merged
        |
        | quantization
        v
<future quantized cybersecurity models>
```

### Original model

```text
models/qwen3-4b-original
```

This is the original/unquantized BF16 reference model and must remain untouched.

### LoRA checkpoint

```text
models/qwen3-4b-lora-r8/checkpoint-3469
```

This is the one-epoch cybersecurity LoRA checkpoint used for evaluation.

### Merged model

```text
models/qwen3-4b-cybersecurity-lora-r8-merged
```

The adapter was merged into the base model with PEFT `merge_and_unload()` to create a standalone BF16 model for inference and subsequent quantization.

This merged model is now the primary input for the next quantization stage.

---

## 5. Successful vLLM Serving

The original Qwen3-4B model successfully reached vLLM startup and API readiness.

Representative server messages included:

```text
APIServer INFO: Started server process
APIServer INFO: Application startup complete.
```

The model endpoint was verified with:

```bash
curl http://localhost:8000/v1/models
```

The model ID returned by the server was:

```text
models/qwen3-4b-original
```

### Important model-ID lesson

The model's `id` is the value that must be supplied to the chat completion request:

```json
{
  "model": "models/qwen3-4b-original"
}
```

The `permission[].id` value is only a permission record identifier and must not be used as the model name.

---

## 6. Qwen3 Thinking Behavior

The project workload is intended to use Qwen3 without visible reasoning output.

An initial vLLM request produced reasoning-style text such as:

```text
Okay, the user is asking...
First, I need to recall...
```

and consumed the available completion budget.

An attempt to disable thinking with the vLLM `chat_template_kwargs.enable_thinking=false` path did not reliably remove the reasoning output in the tested setup.

The Qwen3 `/no_think` switch was then tested successfully.

Example:

```json
{
  "model": "models/qwen3-4b-original",
  "messages": [
    {
      "role": "user",
      "content": "/no_think\nWhat is PagedAttention in one sentence?"
    }
  ],
  "max_tokens": 80,
  "temperature": 0.7
}
```

### Current approach

Use `/no_think` consistently at the application/message construction layer when reasoning output is not desired.

It must be kept consistent across all model comparisons.

---

## 7. FlashInfer / CUDA Startup Investigation

There was initially a vLLM startup failure involving FlashInfer JIT compilation.

Relevant errors included:

```text
EngineCore failed to start
```

and:

```text
No supported CUDA architectures found for major versions [9, 10, 11, 12].
```

The RTX 4090 reports SM89:

```text
(8, 9)
```

The environment was configured with:

```text
FLASHINFER_CUDA_ARCH_LIST=8.9
FLASHINFER_JIT_VERBOSE=1
```

The installed FlashInfer package was:

```text
flashinfer-python 0.6.14
```

The installed `CompilationContext` has architecture handling that can reject some SM89 paths for components explicitly requesting SM90+ support.

Despite the initial problem, the vLLM server later started successfully, so this issue is **not currently blocking serving**.

---

## 8. CUDA Graph Capture and Serving Memory

Successful vLLM initialization included CUDA graph capture:

```text
Capturing CUDA graphs (FULL): 100%|██████████| 35/35
Graph capturing finished in 3 secs, took 0.60 GiB
```

One initialization memory report for the original BF16 model showed approximately:

```text
Free memory: 16.58 / 23.52 GiB
Desired GPU memory utilization: 0.65
Weight usage: 7.56 GiB
Peak activation: 0.16 GiB
Non-torch memory: 0.10 GiB
CUDA graph memory: 0.60 GiB
KV-cache memory in use: 7.48 GiB
```

These values are historical serving observations and should be re-measured systematically for final model comparisons.

---

## 9. Port Conflicts and GPU Process Management

A later vLLM startup attempt reported that the address was already in use, consistent with another service already listening on port 8000.

Useful checks:

```bash
sudo lsof -i :8000
```

or:

```bash
sudo ss -ltnp | grep :8000
```

An alternate vLLM port can be used with:

```bash
vllm serve models/qwen3-4b-original --port 8001
```

Then the API becomes:

```text
http://localhost:8001/v1
```

At one point `nvidia-smi` also showed Ollama processes consuming substantial VRAM. GPU memory usage should therefore be checked before starting vLLM:

```bash
nvidia-smi
```

Ollama and vLLM GPU-memory conflicts are separate from port conflicts.

---

## 10. Cybersecurity Dataset Curation

The cybersecurity SFT corpus uses two sources:

```text
Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset
        +
Cleaned ShareGPT dataset
        |
        v
Combined dataset
        |
        v
Fresh train/validation split
```

### ShareGPT

The cleaned ShareGPT files are:

```text
data/sharegpt_balanced/train.jsonl
data/sharegpt_balanced/validation.jsonl
```

The two files were combined back into a single cleaned ShareGPT pool before the final dataset was created.

### Final dataset construction

The final merge pipeline is:

```text
ShareGPT train
      +
ShareGPT validation
      |
      v
Combined cleaned ShareGPT
      |
      +------------------+
      |                  |
      v                  v
Normalize ShareGPT   Normalize Trendyol
      |                  |
      +--------+---------+
               |
               v
         Merge datasets
               |
               v
       Exact deduplication
               |
               v
       Reproducible shuffle
               |
               v
        Fresh 98/2 split
```

The final dataset layout is:

```text
data/final/
├── train.jsonl
├── validation.jsonl
└── metadata.json
```

The `source` field is preserved for analysis, while the SFT formatting uses `system`, `user`, and `assistant`.

---

## 11. LoRA SFT Configuration

The cybersecurity model was fine-tuned using supervised fine-tuning with LoRA rather than full-model fine-tuning.

### LoRA configuration

```text
LoRA rank             : 8
LoRA alpha            : 16
LoRA dropout          : 0.05
Learning rate         : 2e-4
Batch size            : 1
Gradient accumulation : 16
Max sequence length   : 2048
Precision             : BF16
Epochs completed      : 1
```

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The training script uses the Qwen chat template and masks the system/user prompt portion so that loss is applied to assistant tokens.

### Smoke test

A 20-step smoke test completed successfully before the full run.

Observed smoke-test result:

```text
Steps       : 20
Train loss  : 1.484
Eval loss   : 1.307
Runtime     : 253.5 s
```

The full one-epoch LoRA run also completed successfully and produced the checkpoint:

```text
models/qwen3-4b-lora-r8/checkpoint-3469
```

### Resuming training

The checkpoint can be resumed for another epoch if model-quality evaluation justifies it.

Additional epochs are **not expected to materially improve raw inference tokens/sec**. Training duration and inference throughput are separate concerns.

---

## 12. LoRA Evaluation Results

A deterministic comparison was performed using 11 cybersecurity prompts and a 512-token generation limit.

### Base vs unmerged LoRA

Approximate results:

```text
                         Base BF16       Unmerged LoRA
Average output tokens      511.2             289.6
Average generation time    6.84 s             5.93 s
Average throughput         74.8 tok/s         48.8 tok/s
```

The unmerged LoRA path showed approximately a 35% lower tokens/sec rate in the tested Transformers/PEFT inference setup.

However, the LoRA generated substantially shorter responses, so its average wall-clock generation time was actually slightly lower in this prompt set.

### Response-quality observation

The fine-tuned model demonstrated strong cybersecurity-domain behavior across categories including:

- Network security.
- Identity security / Kerberos.
- Cryptography.
- Web security.
- Secure coding.
- Threat intelligence.
- Incident response.
- Malware analysis.
- Offensive security concepts.
- Cloud security.
- Social engineering.

The model also showed a more direct response style in the tested comparison.

The previous base-vs-LoRA benchmark included visible reasoning behavior in the base output, while the LoRA outputs were much more direct under the tested `/no_think` prompting.

---

## 13. LoRA Merge and Throughput Result

The LoRA adapter was merged into the Qwen3-4B base model using PEFT `merge_and_unload()`.

Merged model:

```text
models/qwen3-4b-cybersecurity-lora-r8-merged
```

### Merged benchmark

The merged model was benchmarked separately.

Observed results:

```text
Average output tokens      : 304.3
Average generation time    : 4.05 s
Average throughput         : 75.0 tok/s
```

This is essentially the same raw token throughput as the original BF16 baseline in the current Transformers benchmark.

### Main conclusion

The approximately 35–40% lower tokens/sec observed with the **unmerged LoRA adapter** is therefore not an inherent property of the fine-tuned model.

It is primarily associated with the separate adapter/runtime path in the tested inference setup.

For production-style serving, the preferred flow is therefore:

```text
Base Qwen3-4B
      +
LoRA adapter
      |
      v
merge_and_unload()
      |
      v
Standalone BF16 cybersecurity model
      |
      v
Quantization / vLLM
```

### Benchmarking warning

A merged model must be benchmarked as a standalone model.

Do **not** apply the same LoRA adapter again on top of an already merged model. Doing so produces an invalid comparison and can result in repeated, nonsensical, or empty outputs.

---

## 14. Previous Pre-Fine-Tuning Quantization Experiment

An earlier quantization experiment was performed on the original Qwen3-4B model before the cybersecurity LoRA stage.

An AWQ quantized model was produced and tested with vLLM/Transformers tooling.

Historical model sizes included approximately:

```text
Original Qwen3-4B model : ~7.6 GB / larger BF16 checkpoint set
AWQ model               : ~3.3 GB
```

A later perplexity benchmark using 128 Wikitext-2 samples at `max_length=1024` reported approximately:

```text
Original perplexity : 22.114871
AWQ perplexity      : 30.815850
```

An AWQ inference benchmark on 10 prompts produced approximately:

```text
Mean latency                 : 0.306 s
Mean generation throughput   : 239.09 tok/s
Total completion tokens      : 734
Total elapsed time           : 3.0617 s
```

These results are useful historical baselines, but they are **not** the final cybersecurity-model quantization results because the AWQ model was generated before the current LoRA fine-tuning stage.

The next quantization experiments must therefore start from:

```text
models/qwen3-4b-cybersecurity-lora-r8-merged
```

rather than the original base model.

---

## 15. Quantization Phase

Quantization is now the next major technical stage.

The key sequencing decision is:

> **Quantize the merged cybersecurity model, not the separate LoRA adapter.**

Target workflow:

```text
Qwen3-4B original BF16
        |
        v
Cybersecurity LoRA SFT
        |
        v
Merged cybersecurity BF16
        |
        +----> W8A16 / GPTQ
        |
        +----> AWQ / additional method
        |
        v
Quantized cybersecurity model
        |
        v
vLLM serving
```

### First planned quantization experiment

The first target is:

```text
W8A16 weight-only quantization
using GPTQ calibration
```

LLM Compressor will be used for the quantization workflow.

The exact recipe should record:

- Quantization algorithm.
- Weight precision.
- Activation precision.
- Calibration dataset.
- Number of calibration samples.
- Calibration sequence length.
- Ignored modules, if any.
- Target hardware.
- Software/package versions.

### Additional experiment

AWQ or another relevant vLLM-supported quantization method can be evaluated afterward for comparison.

The primary comparison should be:

```text
Merged BF16 cybersecurity model
          vs
W8A16/GPTQ cybersecurity model
          vs
Additional quantized candidates
```

---

## 16. Quantization vs Fine-Tuning

These are separate operations with different purposes.

### Quantization

Quantization changes the numerical representation of model parameters and/or activations.

Conceptually:

```text
BF16 / FP16
     |
     v
 INT8 / INT4
```

Typical goals:

- Reduce memory footprint.
- Reduce model storage.
- Improve inference efficiency where supported.
- Preserve as much model quality as possible.

### Fine-tuning

Fine-tuning changes model behavior using training data.

This project uses **LoRA SFT** for cybersecurity specialization.

The current lifecycle is:

```text
Base Qwen3-4B BF16
        |
        v
LoRA cybersecurity SFT
        |
        v
Merged cybersecurity BF16
        |
        v
Quantization
        |
        v
Quantized serving candidate
```

---

## 17. Evaluation and Benchmarking Tools

### LM Evaluation Harness

`lm-eval` is the standardized evaluation tool being used for model-quality comparisons.

It can be used for benchmark tasks such as ARC and other standard accuracy-based evaluations.

Quality comparisons should use identical prompts, settings, and task configurations across model variants.

### Perplexity

Perplexity evaluation has been used with Wikitext-2 to measure language-model degradation after compression.

For the final cybersecurity quantization stage, perplexity should be treated as one metric rather than the sole quality measure.

### Serving benchmarks

vLLM serving should be evaluated using:

- Time to First Token (TTFT).
- End-to-end latency.
- Inter-token latency.
- Output tokens/sec.
- Requests/sec.
- GPU memory.
- KV-cache usage.
- Concurrent request capacity.

Model quality and serving performance are separate evaluation dimensions.

---

## 18. Experimental Design

The complete experiment is now:

```text
Original Qwen3-4B BF16
        |
        +------------------------> Original baseline
        |
        v
Cybersecurity LoRA SFT
        |
        v
Merged cybersecurity BF16
        |
        +------------------------> Fine-tuned BF16 benchmark
        |
        +------------------------> W8A16/GPTQ
        |                              |
        |                              v
        |                       Quantized model
        |
        +------------------------> AWQ / other method
                                       |
                                       v
                                Quantized model(s)
                                       |
                                       v
                                  vLLM serving
                                       |
                                       v
                           Quality + performance comparison
                                       |
                                       v
                             Production recommendation
```

Suggested model structure:

```text
models/
├── qwen3-4b-original/
├── qwen3-4b-lora-r8/
├── qwen3-4b-cybersecurity-lora-r8-merged/
└── <quantized-cybersecurity-models>/
```

The original model must not be overwritten.

---

## 19. Variables to Keep Constant

For valid comparisons, keep as many conditions identical as possible.

### Model

Use the same Qwen3-4B base lineage.

### Prompt set

Use identical prompts.

### Thinking mode

If the intended workload disables reasoning, use:

```text
/no_think
```

consistently.

### Generation parameters

Keep these fixed for a comparison:

```text
temperature
top_p
top_k
max_tokens
sampling mode
```

### Hardware

Use the same RTX 4090.

### Runtime

Keep vLLM and related runtime versions fixed for a given benchmark series.

### Context length

Keep the same context length unless context scaling itself is the experiment.

### Concurrency

Use the same concurrency levels when comparing serving performance.

---

## 20. Metrics to Record

For each model version:

| Category | Metric |
|---|---|
| Model | Model/version |
| Training state | Base / LoRA / merged / quantized |
| Precision | BF16/FP16/INT8/INT4/etc. |
| Quantization method | GPTQ/AWQ/etc. |
| Model disk size | GB |
| GPU | RTX 4090 |
| Peak VRAM | GiB |
| KV cache | Capacity/usage |
| TTFT | ms |
| End-to-end latency | ms |
| Output throughput | tokens/sec |
| Request throughput | requests/sec |
| Concurrency | Tested concurrency |
| Quality | LM-eval / domain metrics |
| Perplexity | Wikitext/domain set |
| Thinking | Enabled/disabled |
| Context | Tested context length |

For the cybersecurity model, response quality should also be manually or automatically evaluated across the relevant domains.

---

## 21. Current Project Status

### Completed

- [x] Local model execution experience with Ollama.
- [x] Decision to move toward vLLM for production serving.
- [x] Python virtual environment established.
- [x] CUDA-enabled PyTorch verified.
- [x] RTX 4090 detected.
- [x] GPU compute capability verified as SM89.
- [x] FlashInfer installed.
- [x] vLLM 0.26.0 installed.
- [x] Original Qwen3-4B model available.
- [x] Original model served through vLLM.
- [x] `/v1/models` endpoint verified.
- [x] `/v1/chat/completions` endpoint verified.
- [x] Incorrect model ID / permission ID issue diagnosed.
- [x] Qwen3 thinking behavior investigated.
- [x] `/no_think` successfully tested.
- [x] CUDA graph capture successfully completed.
- [x] FlashInfer architecture/startup issue investigated.
- [x] Ollama GPU-memory conflict identified.
- [x] ShareGPT dataset cleaned and converted to train/validation JSONL.
- [x] ShareGPT train + validation combined for the final corpus.
- [x] Trendyol + cleaned ShareGPT merged and normalized.
- [x] Final combined train/validation dataset generated.
- [x] LoRA SFT smoke test completed successfully.
- [x] One full LoRA training epoch completed.
- [x] LoRA checkpoint saved.
- [x] LoRA merged into the base BF16 model.
- [x] Merged-model throughput benchmark completed.
- [x] Merged model reached approximately 75 tok/s in the current Transformers benchmark.
- [x] Previous AWQ experiment on the original model completed.
- [x] Previous AWQ perplexity/throughput results recorded.

### In progress / next

- [ ] Decide whether a second LoRA epoch is necessary for quality.
- [ ] Select cybersecurity calibration data.
- [ ] Quantize the merged cybersecurity model with W8A16/GPTQ.
- [ ] Evaluate quantized quality/perplexity.
- [ ] Serve the quantized model with vLLM.
- [ ] Benchmark quantized VRAM, TTFT, latency, throughput, and concurrency.
- [ ] Run LM-eval and cybersecurity-specific evaluation on final candidates.
- [ ] Compare GPTQ against AWQ or another relevant method.
- [ ] Produce the final production recommendation.

---

## 22. Recommended Next Steps

### Step 1 — Preserve the current artifacts

Keep:

```text
models/qwen3-4b-original
models/qwen3-4b-lora-r8/checkpoint-3469
models/qwen3-4b-cybersecurity-lora-r8-merged
```

The original and LoRA checkpoint should remain available for reproducibility and possible additional training.

### Step 2 — Use the merged BF16 model as the quantization reference

Primary input:

```text
models/qwen3-4b-cybersecurity-lora-r8-merged
```

This is the correct model to quantize because it contains the cybersecurity fine-tuning.

### Step 3 — Quantize with LLM Compressor

Start with:

```text
W8A16 + GPTQ
```

Use representative cybersecurity calibration data and record the complete recipe.

### Step 4 — Validate the quantized model

Before serving, verify:

- Model loads correctly.
- Model generates correctly.
- No tokenizer/chat-template regressions.
- No obvious quality collapse.
- Expected model size reduction is achieved.

### Step 5 — Serve with vLLM

Use the same `/no_think` behavior and generation configuration as the BF16 benchmark.

### Step 6 — Re-run the same benchmark suite

Compare:

```text
Merged BF16
      vs
W8A16/GPTQ
      vs
AWQ / additional candidates
```

Measure:

- Quality.
- Perplexity.
- Model disk size.
- VRAM.
- TTFT.
- End-to-end latency.
- Output tokens/sec.
- Request throughput.
- Concurrency.

### Step 7 — Select the production candidate

The best model is not necessarily the smallest model. Select the configuration with the best practical trade-off between cybersecurity quality, GPU memory, latency, throughput, and vLLM compatibility.

---

## 23. Final Goal

The objective is not simply to make Qwen3-4B smaller.

The real goal is to determine:

> **Which cybersecurity-specialized Qwen3-4B configuration provides the best production trade-off between model quality, GPU memory consumption, latency, throughput, and vLLM compatibility on the available RTX 4090 hardware?**

The project should therefore progress as:

```text
Original Qwen3-4B BF16
        |
        v
Original baseline
        |
        v
Cybersecurity LoRA SFT
        |
        v
Merged cybersecurity BF16
        |
        +--------------------+
        |                    |
        v                    v
BF16 benchmark       Quantization experiments
                             |
                             +--> W8A16/GPTQ
                             |
                             +--> AWQ / other method
                             |
                             v
                        vLLM serving
                             |
                             v
                  Quality + performance comparison
                             |
                             v
                    Production recommendation
```

### Current Position

**The project has completed the original serving stage and the first cybersecurity LoRA fine-tuning stage. The LoRA adapter has been merged successfully, and the merged BF16 model has demonstrated approximately base-model token throughput in the current inference benchmark. The next major technical task is to quantize this merged cybersecurity model, beginning with W8A16/GPTQ, and evaluate the quantized model under vLLM.**
