# LLM Quantization & Serving Experiment

## 1. Project Overview

This project is an end-to-end experiment for evaluating and deploying local Large Language Models (LLMs), with a primary focus on:

- Running models locally.
- Moving from Ollama-based local inference to vLLM-based production-style serving.
- Establishing an unquantized/original-model baseline.
- Quantizing the model using modern LLM quantization/compression methods.
- Comparing original and quantized models on quality, memory usage, latency, and serving performance.
- Eventually determining the best production trade-off for the available GPU.

The main model used so far is **Qwen3-4B**.

**Current phase:** baseline serving. The original Qwen3-4B model has been successfully served through vLLM. Quantization and fine-tuning have not yet been performed.

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

System `nvcc` is currently:

```text
CUDA 12.0
V12.0.140
/usr/bin/nvcc
```

This is different from the CUDA 13.0 runtime bundled with the installed PyTorch build and is an important detail for future compiler/extension troubleshooting.

---

## 3. Ollama → vLLM Migration

Ollama was used for earlier local model execution.

The project is now moving toward **vLLM for production-style serving**.

### Ollama

Useful for:

- Easy local model execution.
- Rapid experimentation.
- Simple model management.

### vLLM

The target serving engine because it provides:

- OpenAI-compatible APIs.
- Efficient GPU inference.
- KV-cache management.
- CUDA graph execution.
- Optimized batching/serving behavior.
- A production-oriented inference server.

The current workflow is therefore:

```text
Model
  ↓
vLLM
  ↓
OpenAI-compatible HTTP API
  ↓
Application
```

---

## 4. Model

The currently served model is:

```text
models/qwen3-4b-original
```

This is the **original/unquantized baseline model**.

So far:

- Quantization: **not performed**
- Fine-tuning: **not performed**
- LoRA/QLoRA: **not performed**
- GPTQ: **not performed**
- AWQ: **not performed**
- LLM Compressor quantization: **not performed**
- Calibration: **not performed**

The original model should remain untouched so that it can serve as the baseline for future experiments.

---

## 5. Successful vLLM Serving

The vLLM API server successfully reached:

```text
APIServer INFO: Started server process
APIServer INFO: Application startup complete.
```

The model endpoint was verified with:

```bash
curl http://localhost:8000/v1/models
```

The returned model ID was:

```text
models/qwen3-4b-original
```

### Important model-ID lesson

The model's `id` is the value that must be sent as:

```json
"model": "models/qwen3-4b-original"
```

The `permission[].id` value is only a permission record identifier and must **not** be used as the model name.

---

## 6. Initial Chat Completion Test

A request was sent to:

```text
http://localhost:8000/v1/chat/completions
```

with:

```json
{
  "model": "models/qwen3-4b-original",
  "messages": [
    {
      "role": "user",
      "content": "What is PagedAttention in one sentence?"
    }
  ],
  "max_tokens": 80,
  "temperature": 0.7
}
```

The request successfully reached Qwen3.

However, the model generated reasoning-style text such as:

```text
Okay, the user is asking...
First, I need to recall...
```

and consumed the full 80-token completion budget.

The response ended with:

```text
"finish_reason": "length"
```

This showed that Qwen3 thinking/reasoning mode was active.

---

## 7. Disabling Qwen3 Thinking

The project requirement is to use Qwen3 **without thinking/reasoning output** for the intended workload.

An attempt was made to use:

```json
"extra_body": {
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

but the observed response still contained reasoning-style generation.

The Qwen3 `/no_think` switch was then tested and **worked successfully**.

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

For the current workload, `/no_think` should be included in the application/message construction layer.

For example:

```python
messages = [
    {
        "role": "user",
        "content": f"/no_think\n{user_prompt}"
    }
]
```

This should be used consistently during benchmarks so that original and quantized models are evaluated under the same generation behavior.

---

## 8. FlashInfer / CUDA Startup Investigation

There was initially a vLLM startup failure involving FlashInfer JIT compilation.

The error included:

```text
EngineCore failed to start
```

and a failed Ninja command for FlashInfer's sampling extension.

Another important error was:

```text
No supported CUDA architectures found for major versions [9, 10, 11, 12].
```

### GPU architecture

The RTX 4090 reports:

```text
(8, 9)
```

which is SM89.

The environment was configured with:

```text
FLASHINFER_CUDA_ARCH_LIST=8.9
```

and:

```text
FLASHINFER_JIT_VERBOSE=1
```

### FlashInfer version

```text
flashinfer-python 0.6.14
```

Package location:

```text
/mnt/NewDisk/llm_quant_experiment/.venv/lib/python3.12/site-packages/flashinfer
```

### Relevant FlashInfer architecture behavior

The installed `CompilationContext` normalizes architectures roughly as follows:

- SM9.x → suffix `a`
- SM12.x → architecture-specific suffixes
- SM10+ → suffix `a`
- SM < 9 → no suffix

Therefore SM89 is represented as:

```text
8.9
```

Several FlashInfer components explicitly request:

```python
supported_major_versions=[9, 10, 11, 12]
```

Those components are intended for SM90+ and can reject an SM89 GPU.

Despite the initial issue, the vLLM server subsequently started successfully, so this is **not currently blocking model serving**.

---

## 9. CUDA Graph Capture

Successful vLLM initialization included:

```text
Capturing CUDA graphs (FULL): 100%|██████████| 35/35
```

followed by:

```text
Graph capturing finished in 3 secs, took 0.60 GiB
```

This confirms that the engine successfully reached CUDA graph initialization.

A memory report during initialization showed approximately:

```text
Free memory: 16.58 / 23.52 GiB
Desired GPU memory utilization: 0.65
Weight usage: 7.56 GiB
Peak activation: 0.16 GiB
Non-torch memory: 0.10 GiB
CUDA graph memory: 0.60 GiB
KV-cache memory in use: 7.48 GiB
```

These numbers are useful as an initial serving baseline, although they should be re-measured systematically during benchmarking.

---

## 10. Current Serving Architecture

```text
                    Qwen3-4B Original
                           |
                           v
                    +-------------+
                    |    vLLM     |
                    |   0.26.0    |
                    +-------------+
                           |
                  OpenAI-compatible API
                           |
                           v
               http://localhost:8000/v1
                           |
             +-------------+-------------+
             |                           |
             v                           v
       /v1/models              /v1/chat/completions
```

Application-side architecture:

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

## 11. Port Conflicts

A later vLLM startup attempt reported that the address was already in use.

This is normally a port conflict, especially if an existing vLLM server is already listening on port 8000.

Check:

```bash
sudo lsof -i :8000
```

or:

```bash
sudo ss -ltnp | grep :8000
```

If the existing process is the previous vLLM server, stop it using its PID:

```bash
kill <PID>
```

If necessary:

```bash
kill -9 <PID>
```

Alternatively, use another port:

```bash
vllm serve models/qwen3-4b-original --port 8001
```

Then the API becomes:

```text
http://localhost:8001/v1
```

---

## 12. Ollama GPU Processes

At one point `nvidia-smi` showed two Ollama processes:

```text
/usr/local/bin/ollama   PID 946199   ~1630 MiB
/usr/local/bin/ollama   PID 967238   ~13636 MiB
```

Together these consumed approximately:

```text
15.2 GiB VRAM
```

Before running vLLM, GPU memory should therefore be checked with:

```bash
nvidia-smi
```

The process ownership can be checked with:

```bash
ps -fp 946199
ps -fp 967238
```

If these are no longer needed:

```bash
kill 946199 967238
```

or, if required:

```bash
kill -9 946199 967238
```

If Ollama is running as a system service:

```bash
sudo systemctl stop ollama
```

Then verify:

```bash
nvidia-smi
```

Note that GPU memory usage and port conflicts are separate problems: an Ollama process can consume GPU memory without necessarily owning port 8000.

---

# 13. Quantization Phase

Quantization is the **next major phase**.

The goal is to create one or more quantized versions of Qwen3-4B and compare them against the original model.

Candidate approaches discussed so far include:

- LLM Compressor.
- GPTQ.
- AWQ.
- Other vLLM-supported quantization formats.

LLM Compressor is being considered as a deployment-oriented compression/quantization tool, particularly for workflows involving calibration.

---

## 14. Quantization vs Fine-Tuning

These are separate operations.

### Quantization

Quantization changes the numerical representation of model parameters and/or activations.

Conceptually:

```text
BF16 / FP16
     ↓
 INT8 / INT4
```

Typical goals:

- Reduce memory footprint.
- Reduce model storage.
- Improve inference efficiency where supported.
- Preserve as much model quality as possible.

### Fine-tuning

Fine-tuning changes model parameters using training data.

Examples:

- Full fine-tuning.
- LoRA.
- QLoRA.

Fine-tuning is primarily about changing/adapting model behavior.

### Current status

Neither quantization nor fine-tuning has been performed yet.

---

# 15. Baseline Benchmarking

Before quantization, the original model should be benchmarked.

The baseline should record:

```text
Original Qwen3-4B
        |
        +--> Quality
        +--> VRAM
        +--> TTFT
        +--> Latency
        +--> Tokens/sec
        +--> Request throughput
        +--> Concurrency
```

This baseline becomes the reference for all later quantized models.

---

## 16. Evaluation and Benchmarking Tools

### LM Evaluation Harness

`lm-eval` is intended for standardized model-quality evaluation.

It can be used for benchmark tasks such as:

- ARC.
- Other academic/general-knowledge benchmarks.
- Accuracy-based evaluations.

Quality benchmarking should be run consistently between the original and quantized models.

### Serving benchmarks

vLLM serving should additionally be evaluated using operational metrics such as:

- Time to First Token (TTFT).
- End-to-end latency.
- Inter-token latency.
- Output tokens/second.
- Requests/second.
- GPU memory.
- KV-cache usage.
- Concurrent request capacity.

Model quality and serving performance should be treated as separate dimensions.

---

# 17. Experimental Design

The recommended experiment is:

```text
                    ORIGINAL MODEL
                          |
                          v
                  Baseline Evaluation
                          |
             +------------+------------+
             |                         |
             v                         v
       Quality Metrics          Serving Metrics
                                  |
                                  v
                         Quantization Stage
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
               INT8             INT4           Other scheme
                 |                |                |
                 +----------------+----------------+
                                  |
                                  v
                         Same benchmark suite
                                  |
                                  v
                         Compare to baseline
```

The original model should not be overwritten.

Suggested structure:

```text
models/
├── qwen3-4b-original/
└── qwen3-4b-quantized/
```

---

# 18. Variables to Keep Constant

For valid comparisons, keep as many conditions identical as possible.

### Model

Use the same base Qwen3-4B model.

### Prompt set

Use identical prompts.

### Thinking mode

Use:

```text
/no_think
```

consistently if the production workload disables reasoning.

### Generation parameters

Keep parameters consistent, for example:

```text
temperature
top_p
top_k
max_tokens
```

### Hardware

Use the same RTX 4090.

### vLLM version

Keep vLLM fixed for a given comparison.

### Context length

Keep context settings consistent unless context-length scaling is itself the experiment.

### Concurrency

Use the same concurrency levels.

---

# 19. Metrics to Record

For each model version:

| Category | Metric |
|---|---|
| Model | Model/version |
| Precision | BF16/FP16/INT8/INT4/etc. |
| Quantization method | Method used |
| Model disk size | GB |
| GPU | RTX 4090 |
| Peak VRAM | GiB |
| KV cache | Capacity/usage |
| TTFT | ms |
| End-to-end latency | ms |
| Output throughput | tokens/sec |
| Request throughput | requests/sec |
| Concurrency | Tested concurrency |
| Quality | LM-eval metrics |
| Thinking | Enabled/disabled |
| Context | Tested context length |

---

# 20. Current Project Status

## Completed

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
- [x] Incorrect model ID/permission ID issue diagnosed.
- [x] Qwen3 thinking behavior investigated.
- [x] `/no_think` successfully tested.
- [x] CUDA graph capture successfully completed.
- [x] FlashInfer architecture/startup issue investigated.
- [x] Ollama GPU-memory conflict identified.

## Not yet completed

- [ ] Formal baseline benchmark.
- [ ] Formal LM-eval quality benchmark.
- [ ] Calibration dataset selection.
- [ ] Quantization with LLM Compressor.
- [ ] GPTQ experiment.
- [ ] AWQ experiment.
- [ ] Quantized vLLM serving.
- [ ] Original-vs-quantized quality comparison.
- [ ] Original-vs-quantized latency comparison.
- [ ] Original-vs-quantized throughput comparison.
- [ ] Original-vs-quantized VRAM comparison.
- [ ] Concurrency testing.
- [ ] Production configuration.
- [ ] Final quantization-method recommendation.

---

# 21. Recommended Next Steps

## Step 1 — Clean the environment

Check GPU usage:

```bash
nvidia-smi
```

Check port 8000:

```bash
sudo lsof -i :8000
```

Stop unused Ollama/vLLM processes if necessary.

---

## Step 2 — Establish the original-model baseline

Benchmark:

```text
models/qwen3-4b-original
```

with thinking disabled.

Record:

- TTFT.
- Output tokens/sec.
- End-to-end latency.
- VRAM.
- KV-cache usage.
- Multiple concurrency levels.
- Model quality using LM Evaluation Harness.

---

## Step 3 — Select a quantization method

Start with one deployment-oriented method.

LLM Compressor is a candidate.

Define:

- Quantization format.
- Weight precision.
- Activation precision.
- Calibration dataset.
- Number of calibration samples.
- Calibration sequence length.
- Target GPU/hardware.

---

## Step 4 — Produce the quantized model

Keep the original model untouched:

```text
models/
├── qwen3-4b-original/
└── qwen3-4b-quantized/
```

Document the exact quantization configuration.

---

## Step 5 — Serve the quantized model with vLLM

Verify:

```bash
curl http://localhost:8000/v1/models
```

and:

```bash
curl http://localhost:8000/v1/chat/completions
```

Use the same prompt and generation settings as the baseline.

---

## Step 6 — Re-run the same benchmark suite

Compare:

```text
Original
   vs
Quantized
```

for:

- Quality.
- VRAM.
- TTFT.
- Latency.
- Throughput.
- Concurrency.

---

# 22. Final Goal

The objective is not simply to make Qwen3-4B smaller.

The real goal is to determine:

> **Which model configuration provides the best production trade-off between model quality, GPU memory consumption, latency, and throughput on the available RTX 4090 hardware.**

The overall experiment should therefore progress as:

```text
Original Qwen3-4B
        |
        v
Baseline quality + performance
        |
        v
Quantization
        |
        v
Quantized Qwen3-4B
        |
        v
Same quality + performance tests
        |
        v
Comparison
        |
        v
Production recommendation
```

## Current Position

**The project is currently at the end of the original-model serving stage and before the quantization stage.**

The next major technical task should be to establish a reproducible baseline benchmark for the original Qwen3-4B model before modifying it.
