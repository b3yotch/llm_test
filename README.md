\# LLM Quantization & Serving Experiment

\## 1. Project Overview

This project is an end-to-end experiment for evaluating and deploying local Large Language Models (LLMs), with a primary focus on:

\- Running models locally.

\- Moving from Ollama-based local inference to vLLM-based production-style serving.

\- Establishing an unquantized/original-model baseline.

\- Quantizing the model using modern LLM quantization/compression methods.

\- Comparing original and quantized models on quality, memory usage, latency, and serving performance.

\- Eventually determining the best production trade-off for the available GPU.

The main model used so far is **\*\*Qwen3-4B\*\***.

**\*\*Current phase:\*\*** completed fine-tune → merge → GPTQ quantization → vLLM evaluation. The final W8A16 GPTQ cybersecurity model has been successfully served and benchmarked.

**---**

\## 2. Current Environment

\### Hardware

\- GPU: **\*\*NVIDIA GeForce RTX 4090\*\***

\- GPU capacity reported by PyTorch/vLLM: approximately **\*\*23.52 GiB\*\***

\- CUDA compute capability: **\*\*8.9 / SM89\*\***

Verified with:

\`\`\`bash

python -c "import torch; print(torch.cuda.get\_device\_capability(0))"

\`\`\`

Output:

\`\`\`text

(8, 9)

\`\`\`

\### Software

\`\`\`text

Python: 3.12

PyTorch: 2.11.0+cu130

PyTorch CUDA runtime: 13.0

vLLM: 0.26.0

FlashInfer: 0.6.14

\`\`\`

Project virtual environment:

\`\`\`text

/mnt/NewDisk/llm\_quant\_experiment/.venv

\`\`\`

Project directory:

\`\`\`text

/mnt/NewDisk/llm\_quant\_experiment

\`\`\`

System \`nvcc\` is currently:

\`\`\`text

CUDA 12.0

V12.0.140

/usr/bin/nvcc

\`\`\`

This is different from the CUDA 13.0 runtime bundled with the installed PyTorch build and is an important detail for future compiler/extension troubleshooting.

**---**

\## 3. Ollama → vLLM Migration

Ollama was used for earlier local model execution.

The project is now moving toward **\*\*vLLM for production-style serving\*\***.

\### Ollama

Useful for:

\- Easy local model execution.

\- Rapid experimentation.

\- Simple model management.

\### vLLM

The target serving engine because it provides:

\- OpenAI-compatible APIs.

\- Efficient GPU inference.

\- KV-cache management.

\- CUDA graph execution.

\- Optimized batching/serving behavior.

\- A production-oriented inference server.

The current workflow is therefore:

\`\`\`text

Model

  ↓

vLLM

  ↓

OpenAI-compatible HTTP API

  ↓

Application

\`\`\`

**---**

\## 4. Model

The original model remains the immutable baseline:

\`\`\`text
models/qwen3-4b-original
\`\`\`

The completed experiment now also contains:

\`\`\`text
models/qwen3-4b-lora-r8/checkpoint-3469
models/qwen3-4b-cybersecurity-lora-r8-merged
models/gptq/qwen3-4b-cybersecurity-lora-r8-w8a16-gptq
\`\`\`

Current model status:

- Original Qwen3-4B BF16 baseline: **complete**
- Cybersecurity LoRA SFT: **complete**
- LoRA merge into standalone BF16: **complete**
- GPTQ W8A16: **complete**
- Quantized vLLM serving: **complete**
- Perplexity evaluation: **complete**
- LM Evaluation Harness: **complete**
- Cybersecurity quality evaluation: **complete**

The original model remains untouched for all baseline comparisons.

**---**

\## 5. Successful vLLM Serving

The vLLM API server successfully reached:

\`\`\`text

APIServer INFO: Started server process

APIServer INFO: Application startup complete.

\`\`\`

The model endpoint was verified with:

\`\`\`bash

curl http\://localhost:8000/v1/models

\`\`\`

The returned model ID was:

\`\`\`text

models/qwen3-4b-original

\`\`\`

\### Important model-ID lesson

The model's \`id\` is the value that must be sent as:

\`\`\`json

"model": "models/qwen3-4b-original"

\`\`\`

The \`permission[].id\` value is only a permission record identifier and must **\*\*not\*\*** be used as the model name.

**---**

\## 6. Initial Chat Completion Test

A request was sent to:

\`\`\`text

http\://localhost:8000/v1/chat/completions

\`\`\`

with:

\`\`\`json

{

  "model": "models/qwen3-4b-original",

  "messages": [

    {

      "role": "user",

      "content": "What is PagedAttention in one sentence?"

    }

  ],

  "max\_tokens": 80,

  "temperature": 0.7

}

\`\`\`

The request successfully reached Qwen3.

However, the model generated reasoning-style text such as:

\`\`\`text

Okay, the user is asking...

First, I need to recall...

\`\`\`

and consumed the full 80-token completion budget.

The response ended with:

\`\`\`text

"finish\_reason": "length"

\`\`\`

This showed that Qwen3 thinking/reasoning mode was active.

**---**

\## 7. Disabling Qwen3 Thinking

The project requirement is to use Qwen3 **\*\*without thinking/reasoning output\*\*** for the intended workload.

An attempt was made to use:

\`\`\`json

"extra\_body": {

  "chat\_template\_kwargs": {

    "enable\_thinking": false

  }

}

\`\`\`

but the observed response still contained reasoning-style generation.

The Qwen3 \`/no\_think\` switch was then tested and **\*\*worked successfully\*\***.

Example:

\`\`\`json

{

  "model": "models/qwen3-4b-original",

  "messages": [

    {

      "role": "user",

      "content": "/no\_think\nWhat is PagedAttention in one sentence?"

    }

  ],

  "max\_tokens": 80,

  "temperature": 0.7

}

\`\`\`

\### Current approach

For the current workload, \`/no\_think\` should be included in the application/message construction layer.

For example:

\`\`\`python

messages = [

    {

        "role": "user",

        "content": f"/no\_think\n{user\_prompt}"

    }

]

\`\`\`

This should be used consistently during benchmarks so that original and quantized models are evaluated under the same generation behavior.

**---**

\## 8. FlashInfer / CUDA Startup Investigation

There was initially a vLLM startup failure involving FlashInfer JIT compilation.

The error included:

\`\`\`text

EngineCore failed to start

\`\`\`

and a failed Ninja command for FlashInfer's sampling extension.

Another important error was:

\`\`\`text

No supported CUDA architectures found for major versions [9, 10, 11, 12].

\`\`\`

\### GPU architecture

The RTX 4090 reports:

\`\`\`text

(8, 9)

\`\`\`

which is SM89.

The environment was configured with:

\`\`\`text

FLASHINFER\_CUDA\_ARCH\_LIST=8.9

\`\`\`

and:

\`\`\`text

FLASHINFER\_JIT\_VERBOSE=1

\`\`\`

\### FlashInfer version

\`\`\`text

flashinfer-python 0.6.14

\`\`\`

Package location:

\`\`\`text

/mnt/NewDisk/llm\_quant\_experiment/.venv/lib/python3.12/site-packages/flashinfer

\`\`\`

\### Relevant FlashInfer architecture behavior

The installed \`CompilationContext\` normalizes architectures roughly as follows:

\- SM9.x → suffix \`a\`

\- SM12.x → architecture-specific suffixes

\- SM10+ → suffix \`a\`

\- SM < 9 → no suffix

Therefore SM89 is represented as:

\`\`\`text

8.9

\`\`\`

Several FlashInfer components explicitly request:

\`\`\`python

supported\_major\_versions=[9, 10, 11, 12]

\`\`\`

Those components are intended for SM90+ and can reject an SM89 GPU.

Despite the initial issue, the vLLM server subsequently started successfully, so this is **\*\*not currently blocking model serving\*\***.

**---**

\## 9. CUDA Graph Capture

Successful vLLM initialization included:

\`\`\`text

Capturing CUDA graphs (FULL): 100%|██████████| 35/35

\`\`\`

followed by:

\`\`\`text

Graph capturing finished in 3 secs, took 0.60 GiB

\`\`\`

This confirms that the engine successfully reached CUDA graph initialization.

A memory report during initialization showed approximately:

\`\`\`text

Free memory: 16.58 / 23.52 GiB

Desired GPU memory utilization: 0.65

Weight usage: 7.56 GiB

Peak activation: 0.16 GiB

Non-torch memory: 0.10 GiB

CUDA graph memory: 0.60 GiB

KV-cache memory in use: 7.48 GiB

\`\`\`

These numbers are useful as an initial serving baseline, although they should be re-measured systematically during benchmarking.

**---**

\## 10. Current Serving Architecture

\`\`\`text

                    Qwen3-4B Original

                           |

                           v

                    +-------------+

                    |    vLLM     |

                    |   0.26.0    |

                    +-------------+

                           |

                  OpenAI-compatible API

                           |

                           v

               http\://localhost:8000/v1

                           |

             +-------------+-------------+

             |                           |

             v                           v

       /v1/models              /v1/chat/completions

\`\`\`

Application-side architecture:

\`\`\`text

Application

    |

    | OpenAI-compatible HTTP API

    v

vLLM

    |

    v

Qwen3-4B

\`\`\`

**---**

\## 11. Port Conflicts

A later vLLM startup attempt reported that the address was already in use.

This is normally a port conflict, especially if an existing vLLM server is already listening on port 8000.

Check:

\`\`\`bash

sudo lsof -i :8000

\`\`\`

or:

\`\`\`bash

sudo ss -ltnp | grep :8000

\`\`\`

If the existing process is the previous vLLM server, stop it using its PID:

\`\`\`bash

kill \<PID>

\`\`\`

If necessary:

\`\`\`bash

kill -9 \<PID>

\`\`\`

Alternatively, use another port:

\`\`\`bash

vllm serve models/qwen3-4b-original --port 8001

\`\`\`

Then the API becomes:

\`\`\`text

http\://localhost:8001/v1

\`\`\`

**---**

\## 12. Ollama GPU Processes

At one point \`nvidia-smi\` showed two Ollama processes:

\`\`\`text

/usr/local/bin/ollama   PID 946199   \~1630 MiB

/usr/local/bin/ollama   PID 967238   \~13636 MiB

\`\`\`

Together these consumed approximately:

\`\`\`text

15.2 GiB VRAM

\`\`\`

Before running vLLM, GPU memory should therefore be checked with:

\`\`\`bash

nvidia-smi

\`\`\`

The process ownership can be checked with:

\`\`\`bash

ps -fp 946199

ps -fp 967238

\`\`\`

If these are no longer needed:

\`\`\`bash

kill 946199 967238

\`\`\`

or, if required:

\`\`\`bash

kill -9 946199 967238

\`\`\`

If Ollama is running as a system service:

\`\`\`bash

sudo systemctl stop ollama

\`\`\`

Then verify:

\`\`\`bash

nvidia-smi

\`\`\`

Note that GPU memory usage and port conflicts are separate problems: an Ollama process can consume GPU memory without necessarily owning port 8000.

**---**

\# 13. Quantization Phase

Quantization is the **\*\*next major phase\*\***.

The goal is to create one or more quantized versions of Qwen3-4B and compare them against the original model.

Candidate approaches discussed so far include:

\- LLM Compressor.

\- GPTQ.

\- AWQ.

\- Other vLLM-supported quantization formats.

LLM Compressor is being considered as a deployment-oriented compression/quantization tool, particularly for workflows involving calibration.

**---**

\## 14. Quantization vs Fine-Tuning

These are separate operations.

\### Quantization

Quantization changes the numerical representation of model parameters and/or activations.

Conceptually:

\`\`\`text

BF16 / FP16

     ↓

 INT8 / INT4

\`\`\`

Typical goals:

\- Reduce memory footprint.

\- Reduce model storage.

\- Improve inference efficiency where supported.

\- Preserve as much model quality as possible.

\### Fine-tuning

Fine-tuning changes model parameters using training data.

Examples:

\- Full fine-tuning.

\- LoRA.

\- QLoRA.

Fine-tuning is primarily about changing/adapting model behavior.

\### Current status

Fine-tuning and quantization are now complete for the current experiment. The LoRA adapter was trained on the combined cybersecurity SFT dataset, merged into the base model, then quantized with GPTQ W8A16 using WikiText-2 calibration. The quantized model was subsequently served through vLLM and evaluated for perplexity, LM-eval performance, serving latency/throughput, and cybersecurity response quality.

**---**

\# 15. Baseline Benchmarking

Before quantization, the original model should be benchmarked.

The baseline should record:

\`\`\`text

Original Qwen3-4B

        |

        +--> Quality

        +--> VRAM

        +--> TTFT

        +--> Latency

        +--> Tokens/sec

        +--> Request throughput

        +--> Concurrency

\`\`\`

This baseline becomes the reference for all later quantized models.

**---**

\## 16. Evaluation and Benchmarking Tools

\### LM Evaluation Harness

\`lm-eval\` is intended for standardized model-quality evaluation.

It can be used for benchmark tasks such as:

\- ARC.

\- Other academic/general-knowledge benchmarks.

\- Accuracy-based evaluations.

Quality benchmarking should be run consistently between the original and quantized models.

\### Serving benchmarks

vLLM serving should additionally be evaluated using operational metrics such as:

\- Time to First Token (TTFT).

\- End-to-end latency.

\- Inter-token latency.

\- Output tokens/second.

\- Requests/second.

\- GPU memory.

\- KV-cache usage.

\- Concurrent request capacity.

Model quality and serving performance should be treated as separate dimensions.

**---**

\# 17. Experimental Design

The recommended experiment is:

\`\`\`text

                    ORIGINAL MODEL

                          |

                          v

                  Baseline Evaluation

                          |

             +------------+------------+

             |                         |

             v                         v

       Quality Metrics          Serving Metrics

                                  |

                                  v

                         Quantization Stage

                                  |

                 +----------------+----------------+

                 |                |                |

                 v                v                v

               INT8             INT4           Other scheme

                 |                |                |

                 +----------------+----------------+

                                  |

                                  v

                         Same benchmark suite

                                  |

                                  v

                         Compare to baseline

\`\`\`

The original model should not be overwritten.

Suggested structure:

\`\`\`text

models/

├── qwen3-4b-original/

└── qwen3-4b-quantized/

\`\`\`

**---**

\# 18. Variables to Keep Constant

For valid comparisons, keep as many conditions identical as possible.

\### Model

Use the same base Qwen3-4B model.

\### Prompt set

Use identical prompts.

\### Thinking mode

Use:

\`\`\`text

/no\_think

\`\`\`

consistently if the production workload disables reasoning.

\### Generation parameters

Keep parameters consistent, for example:

\`\`\`text

temperature

top\_p

top\_k

max\_tokens

\`\`\`

\### Hardware

Use the same RTX 4090.

\### vLLM version

Keep vLLM fixed for a given comparison.

\### Context length

Keep context settings consistent unless context-length scaling is itself the experiment.

\### Concurrency

Use the same concurrency levels.

**---**

\# 19. Metrics to Record

For each model version:

\| Category | Metric |

\|---|---|

\| Model | Model/version |

\| Precision | BF16/FP16/INT8/INT4/etc. |

\| Quantization method | Method used |

\| Model disk size | GB |

\| GPU | RTX 4090 |

\| Peak VRAM | GiB |

\| KV cache | Capacity/usage |

\| TTFT | ms |

\| End-to-end latency | ms |

\| Output throughput | tokens/sec |

\| Request throughput | requests/sec |

\| Concurrency | Tested concurrency |

\| Quality | LM-eval metrics |

\| Thinking | Enabled/disabled |

\| Context | Tested context length |

**---**

\# 20. Current Project Status

## Completed

- [x] Local model execution experience with Ollama.
- [x] Decision to move toward vLLM for production-style serving.
- [x] Python virtual environment established.
- [x] CUDA-enabled PyTorch verified.
- [x] RTX 4090 detected.
- [x] GPU compute capability verified as SM89.
- [x] FlashInfer installed and startup behavior investigated.
- [x] vLLM 0.26.0 installed.
- [x] Original Qwen3-4B model available and retained as baseline.
- [x] Original model served through vLLM.
- [x] `/v1/models` endpoint verified.
- [x] `/v1/chat/completions` endpoint verified.
- [x] Qwen3 thinking behavior investigated.
- [x] `/no_think` successfully tested.
- [x] CUDA graph capture successfully completed.
- [x] Ollama GPU-memory conflict identified.
- [x] ShareGPT cleaned into train/validation JSONL.
- [x] ShareGPT train + validation recombined for the final SFT pool.
- [x] Trendyol normalized and merged with ShareGPT.
- [x] Final combined dataset split into train/validation.
- [x] LoRA smoke test completed successfully.
- [x] One-epoch cybersecurity LoRA training completed.
- [x] LoRA adapter quality evaluation completed.
- [x] LoRA adapter merged into standalone BF16 model.
- [x] Merged BF16 serving benchmark completed.
- [x] GPTQ W8A16 quantization completed with LLM Compressor 0.12.0.1.
- [x] WikiText-2 calibration used for GPTQ.
- [x] GPTQ W8A16 artifact verified as compressed-tensors `pack-quantized` format.
- [x] GPTQ W8A16 successfully served through vLLM 0.26.0.
- [x] BF16-vs-GPTQ latency/throughput benchmark completed.
- [x] Perplexity comparison completed.
- [x] LM Evaluation Harness comparison completed.
- [x] Three-way cybersecurity quality comparison completed.

## Remaining / optional future work

- [ ] Systematic concurrent-load benchmark.
- [ ] TTFT/inter-token-latency benchmark under concurrency.
- [ ] Detailed VRAM and KV-cache comparison across final candidates.
- [ ] Optional AWQ comparison on the merged cybersecurity model.
- [ ] Final production hardening, monitoring, and deployment configuration.

**---**

# 21. Experimental Progress

The project has progressed through the following validated stages:

```text
Original Qwen3-4B BF16
        ↓
Cybersecurity dataset curation
        ↓
LoRA SFT
        ↓
Merged cybersecurity BF16
        ↓
GPTQ W8A16
        ↓
vLLM 0.26.0
        ↓
Perplexity + LM-eval + quality
        ↓
Latency + throughput comparison
```

The main production candidate is now:

```text
models/gptq/qwen3-4b-cybersecurity-lora-r8-w8a16-gptq
```

**---**

# 22. Final Goal and Current Conclusion

The objective was to determine which configuration provides the best production trade-off between cybersecurity quality, model storage, GPU memory, latency, and throughput on the RTX 4090.

The current evidence favors **GPTQ W8A16 applied to the merged cybersecurity LoRA model**.

```text
Merged BF16
    ↓
7.6 GB
~104 tok/s
15.696882 PPL

GPTQ W8A16
    ↓
4.2 GB
~163 tok/s
15.711576 PPL
```

This corresponds to approximately:

```text
~45% smaller model storage
~57% higher measured generation throughput
~36% lower mean serving latency
~0.094% relative perplexity increase
```

The LM-eval scores were effectively unchanged within the reported uncertainty, and the cybersecurity quality evaluation showed that GPTQ preserved the fine-tuned domain behavior.

**Current conclusion:** the main end-to-end experiment is complete. Remaining work is optional production hardening rather than another required training or quantization stage.

**---**
# 23. Cybersecurity Fine-Tuning Phase

The project was extended from pure quantization/serving into a domain-specialization experiment for cybersecurity.

## 23.1 Training data

Two sources were used:

- **Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset**
- **ChaoticNeutrals/Cybersecurity-ShareGPT**, which was cleaned separately before the final merge

The cleaned ShareGPT files were:

```text
data/sharegpt_balanced/train.jsonl
data/sharegpt_balanced/validation.jsonl
```

For the final SFT dataset, the cleaned ShareGPT train and validation files were intentionally recombined into a single ShareGPT pool. That pool was normalized and merged with Trendyol, followed by exact deduplication and a fresh final train/validation split.

Final dataset structure:

```text
data/final/
├── train.jsonl
├── validation.jsonl
└── metadata.json
```

The common record schema is:

```json
{
  "system": "...",
  "user": "...",
  "assistant": "...",
  "source": "..."
}
```

The dataset pipeline deliberately retained broad cybersecurity coverage rather than treating all offensive-security terminology as noise. Structural/data-quality validation and exact deduplication were applied in the final merge pipeline.

## 23.2 LoRA SFT configuration

The Qwen3-4B model was fine-tuned with LoRA using:

```text
LoRA rank       : 8
LoRA alpha      : 16
LoRA dropout    : 0.05
Bias            : none
Task            : CAUSAL_LM
Learning rate   : 2e-4
Batch size      : 1
Grad accumulation: 16
Max length     : 2048
Precision      : BF16
Epochs         : 1 completed
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

A smoke test of 20 steps completed successfully before the full one-epoch training run.

The trained LoRA checkpoint used for subsequent merging/evaluation was:

```text
models/qwen3-4b-lora-r8/checkpoint-3469
```

## 23.3 LoRA quality observations

The fine-tuned model demonstrated clear cybersecurity specialization across networking, identity/Active Directory, cryptography, web security, secure coding, threat intelligence, incident response, malware analysis, offensive security, cloud security, and social engineering.

One important observation from early evaluation was that the unmerged LoRA model had lower raw generation tokens/sec than the merged BF16 model. The adapter itself, however, produced useful cybersecurity responses and did not require a change in LoRA rank for the current experiment.

---

# 24. LoRA Merge and BF16 Production Candidate

The trained adapter was merged back into the base Qwen3-4B model using PEFT `merge_and_unload`, producing a standalone BF16 model:

```text
models/qwen3-4b-cybersecurity-lora-r8-merged
```

Disk size:

```text
7.6 GB
```

This merged model became the main production candidate before quantization.

A critical inference finding was that the unmerged adapter showed substantially lower measured tokens/sec than the merged model, while the merged model returned to approximately the base-model token throughput. This established that the adapter-serving overhead was not inherent to the learned model weights and that merging is preferable for the final standalone serving artifact.

Measured merged BF16 serving benchmark:

```text
Mean latency                  : ~2.010 s
Median latency                : ~2.355 s
Mean generation throughput    : ~104.02 tok/s
Total completion tokens       : 2095
Total elapsed time            : ~20.10 s
```

---

# 25. GPTQ W8A16 Quantization of the Merged Cybersecurity Model

The merged BF16 cybersecurity model was quantized with **LLM Compressor 0.12.0.1** using the GPTQ modifier.

Input model:

```text
models/qwen3-4b-cybersecurity-lora-r8-merged
```

Output model:

```text
models/gptq/qwen3-4b-cybersecurity-lora-r8-w8a16-gptq
```

## 25.1 GPTQ configuration

The installed LLM Compressor version did not accept the string preset `W8A16_ASYM`, so the W8A16 configuration was expressed explicitly through `config_groups`.

Effective configuration:

```text
Weights            : INT8
Activations        : 16-bit / unquantized activation path
Weight strategy    : group
Group size         : 128
Weight symmetry    : symmetric
Block size         : 128
Dampening fraction : 0.01
Actorder           : static
Ignored module     : lm_head
```

## 25.2 Calibration

WikiText-2 was used as the calibration dataset:

```text
Dataset              : wikitext
Config               : wikitext-2-raw-v1
Calibration samples  : 128
Sequence length      : 1024
Batch size            : 1
```

## 25.3 Quantized artifact verification

The resulting model was verified to contain a compressed-tensors quantization configuration:

```text
quant_method         : compressed-tensors
quantization_status  : compressed
format               : pack-quantized
num_bits             : 8
strategy              : group
group_size           : 128
symmetric            : true
targets               : Linear
ignore                : lm_head
```

Model size comparison:

```text
Merged BF16          : 7.6 GB
GPTQ W8A16           : 4.2 GB
```

This represents approximately a **45% reduction in model disk size**.

---

# 26. GPTQ W8A16 vLLM Serving

The quantized model was successfully loaded by the existing serving environment:

```text
vLLM               : 0.26.0
compressed-tensors : 0.17.1
GPU                 : RTX 4090
```

The GPTQ W8A16 model was successfully served through the OpenAI-compatible vLLM API and returned valid cybersecurity responses.

Example model ID:

```text
models/gptq/qwen3-4b-cybersecurity-lora-r8-w8a16-gptq
```

The successful API test confirmed the complete path:

```text
Merged BF16
    ↓
GPTQ W8A16
    ↓
compressed-tensors
    ↓
vLLM 0.26.0
    ↓
RTX 4090
    ↓
Successful inference
```

---

# 27. Final Serving Benchmark

The same 10 cybersecurity/general prompts were benchmarked against the merged BF16 model and GPTQ W8A16 model with deterministic generation.

## 27.1 Merged BF16

```text
Mean latency                  : 2.0101 s
Median latency                : 2.3546 s
Mean generation throughput    : 104.02 tok/s
Total completion tokens       : 2095
Total elapsed time            : 20.1015 s
```

## 27.2 GPTQ W8A16

```text
Mean latency                  : 1.2867 s
Median latency                : 1.5626 s
Mean generation throughput    : 163.13 tok/s
Total completion tokens       : 2103
Total elapsed time            : 12.8666 s
```

## 27.3 Measured improvement

GPTQ W8A16 delivered approximately:

```text
Model size reduction          : ~45%
Mean latency reduction        : ~36%
Generation throughput increase: ~57%
```

Because the completion token counts were nearly identical (2095 vs 2103), the throughput improvement is not explained by substantially shorter outputs.

---

# 28. Perplexity Evaluation

Perplexity was measured on the same WikiText-2 test corpus using:

```text
Samples     : 128-class evaluation configuration used in the project
Max length  : 1024
Tokens      : 13,359
```

Results:

| Model | Perplexity |
|---|---:|
| Merged BF16 | **15.696882** |
| GPTQ W8A16 | **15.711576** |

Absolute change:

```text
+0.014694 perplexity
```

Relative change:

```text
~+0.094%
```

The W8A16 GPTQ model therefore showed **negligible perplexity degradation** in this test.

---

# 29. LM Evaluation Harness Results

The same zero-shot evaluation settings were used for the merged BF16 and GPTQ W8A16 models.

## 29.1 ARC-Challenge

| Metric | Merged BF16 | GPTQ W8A16 |
|---|---:|---:|
| acc | 0.4923 | 0.4940 |
| acc_norm | 0.5188 | 0.5213 |

The differences are smaller than the reported standard errors and should be treated as effectively equivalent.

## 29.2 HellaSwag

| Metric | Merged BF16 | GPTQ W8A16 |
|---|---:|---:|
| acc | 0.5254 | 0.5251 |
| acc_norm | 0.6971 | 0.6960 |

Again, the differences are very small relative to the reported uncertainty.

## 29.3 Interpretation

The measured LM-eval results indicate that **GPTQ W8A16 preserved the tested general-language benchmark quality of the merged BF16 model**.

---

# 30. Final Cybersecurity Quality Evaluation

A final three-way qualitative/API evaluation was performed across 11 cybersecurity prompts covering:

```text
network security
identity security
cryptography
web security
secure coding
threat intelligence
incident response
malware analysis
offensive security
cloud security
social engineering
```

Models compared:

```text
1. Original Qwen3-4B
2. Merged cybersecurity BF16
3. GPTQ W8A16 cybersecurity
```

Evaluation settings:

```text
max_tokens       : 512
temperature      : 0.0
/no_think        : enabled
```

## 30.1 Automated quality indicators

| Metric | Original | Merged BF16 | GPTQ W8A16 |
|---|---:|---:|---:|
| Examples | 11 | 11 | 11 |
| Empty responses | 0 | 0 | 0 |
| Truncated responses | 0 | 0 | 1 |
| Mean completion tokens | 270.36 | 339.64 | 354.27 |
| Mean repetition score | 0.0071 | 0.0075 | **0.0022** |
| Visible reasoning markers | 0 | 0 | 0 |

All three models produced empty `<think></think>` wrappers under the tested Qwen3 chat-template behavior, but no visible reasoning content was detected.

## 30.2 Cybersecurity specialization

The merged BF16 model produced more detailed domain-focused cybersecurity responses than the original model across the evaluation categories.

The GPTQ W8A16 model retained that specialization rather than reverting toward the original model behavior. Its responses remained coherent across networking, Kerberos/Active Directory, cryptography, SQL injection, secure coding, threat intelligence, incident response, malware analysis, vulnerability assessment, cloud security, and phishing/social-engineering topics.

One GPTQ response (IOC/TTP) reached the 512-token limit and was therefore flagged as truncated. This was treated as a generation-length/style issue rather than evidence of quantization failure.

---

# 31. Final Model Comparison

The current experiment can now be summarized as:

| Metric | Merged BF16 | GPTQ W8A16 |
|---|---:|---:|
| Model size | 7.6 GB | **4.2 GB** |
| Perplexity | 15.696882 | **15.711576** |
| Mean serving latency | 2.010 s | **1.287 s** |
| Mean generation throughput | 104.02 tok/s | **163.13 tok/s** |
| ARC acc | 0.4923 | 0.4940 |
| ARC acc_norm | 0.5188 | 0.5213 |
| HellaSwag acc | 0.5254 | 0.5251 |
| HellaSwag acc_norm | 0.6971 | 0.6960 |
| Empty quality responses | 0 | 0 |
| Truncated quality responses | 0 | 1 |
| Mean repetition score | 0.0075 | **0.0022** |

The current evidence supports selecting the GPTQ W8A16 model as the primary deployment candidate for this experiment because it provides a large storage and serving-efficiency improvement with negligible measured general-language quality loss and preserved cybersecurity behavior.

---

# 32. Final Artifact Locations

```text
models/
├── qwen3-4b-original/
├── qwen3-4b-cybersecurity-lora-r8/
│   └── checkpoint-3469/
├── qwen3-4b-cybersecurity-lora-r8-merged/
└── gptq/
    └── qwen3-4b-cybersecurity-lora-r8-w8a16-gptq/
```

Training data:

```text
data/final/
├── train.jsonl
├── validation.jsonl
└── metadata.json
```

Evaluation outputs include the serving benchmark JSON files, perplexity results, LM-eval outputs, and the final three-way cybersecurity quality comparison.

---

# 33. Final Project Position

**The main end-to-end experiment is complete.**

The final pipeline is:

```text
Qwen3-4B BF16
      ↓
Cybersecurity dataset curation
      ↓
LoRA SFT
      ↓
Merged cybersecurity BF16
      ↓
GPTQ W8A16 using WikiText-2 calibration
      ↓
vLLM 0.26.0 serving
      ↓
Perplexity + LM-eval + cybersecurity quality
      ↓
Latency + throughput benchmarking
```

The current preferred model is:

```text
models/gptq/qwen3-4b-cybersecurity-lora-r8-w8a16-gptq
```

The main optional future work is production-hardening: systematic concurrent-load testing, TTFT/inter-token latency under load, detailed VRAM/KV-cache measurement, and optionally comparing another quantization method such as AWQ on the same merged cybersecurity model.
