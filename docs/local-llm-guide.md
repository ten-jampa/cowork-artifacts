# Local LLM Guide — Apple M4 Pro 24 GB

> **Hardware:** M4 Pro · 12-core CPU · 16 GPU cores · 24 GB unified memory · 273 GB/s bandwidth · Metal 4  
> **Verified:** LM Studio 0.4+ · MLX · Ollama · April 2026

Running large language models locally on Apple Silicon — what fits, how fast it runs, and whether upgrading hardware is worth it.

---

## Table of Contents

1. [Your Machine at a Glance](#your-machine-at-a-glance)
2. [How Apple Silicon Changes Everything](#how-apple-silicon-changes-everything)
3. [Model Landscape](#model-landscape)
4. [Speed Reference](#speed-reference)
5. [Setup — LM Studio & Inference Tools](#setup-lm-studio-inference-tools)
6. [Monitoring Memory & Performance](#monitoring-memory-performance)
7. [Hardware Upgrade Analysis — M4 Pro vs M5 Max](#hardware-upgrade-analysis-m4-pro-vs-m5-max)
8. [eGPU — Why External NVIDIA GPUs Don't Work on Mac](#egpu-why-external-nvidia-gpus-dont-work-on-mac)
9. [Verdict & Action Plan](#verdict-action-plan)

---

## Your Machine at a Glance

| Spec | Value |
|---|---|
| Chip | Apple M4 Pro |
| CPU | 12-core (8 performance + 4 efficiency) |
| GPU | 16-core |
| ANE | 16-core Neural Engine |
| Unified Memory | 24 GB |
| Memory Bandwidth | 273 GB/s |
| GPU API | Metal 4 |
| Effective LLM Budget | ~20 GB (after ~4 GB macOS overhead) |

---

## How Apple Silicon Changes Everything

Apple Silicon uses **Unified Memory Architecture (UMA)**: the CPU, GPU, and Neural Engine all share a single pool of high-bandwidth memory. There is no separate VRAM — the full 24 GB is available to the GPU for inference.

This is different from a discrete GPU on a PC: a typical 16 GB laptop GPU has 16 GB VRAM; your M4 Pro effectively has 20 GB available to the GPU (after macOS overhead). That's competitive — without the discrete GPU cost or power draw.

### Inference Speed Formula

Tokens per second on Apple Silicon is driven almost entirely by **memory bandwidth**, not compute:

```
theoretical max tok/s ≈ memory_bandwidth_GB/s ÷ model_size_in_GB
```

For your M4 Pro (273 GB/s):
- 8 GB model → ~34 tok/s theoretical ceiling
- 16 GB model → ~17 tok/s
- 20 GB model → ~14 tok/s

Real-world speeds are 60–80% of theoretical due to framework overhead, quantization, attention computation etc.

### MoE — A Different Speed Formula

Mixture-of-Experts (MoE) models like Qwen3-Coder-30B-A3B and Gemma 4 26B break the pattern:

- All expert weights must be in RAM (same total footprint as an equivalent dense model)
- But only **active parameters** are read per token
- Speed ≈ bandwidth ÷ **active_params_size**, not total size

So Qwen3-Coder-30B-A3B (30B total, 3B active) runs at speeds comparable to a 3B dense model, while fitting 30B-class knowledge into 24 GB.

---

## Model Landscape

### What Fits in 24 GB (20 GB effective)

VRAM estimate formula for Q4_K_M quantization:
```
GB ≈ params_B × 4.5 / 8
```

| Model | Params | Size (Q4_K_M) | Est. Speed | Type | SWE-bench | Best For |
|---|---|---|---|---|---|---|
| Llama 3.2 3B | 3B | 1.9 GB | ~90 tok/s | Dense | — | Fast answers, low latency |
| Phi-4 Mini | 3.8B | 2.3 GB | ~80 tok/s | Dense | — | Compact reasoning |
| Phi-4 14B | 14B | 8.5 GB | ~33 tok/s | Dense | — | Math, reasoning, STEM |
| Qwen2.5-Coder 14B | 14B | 8.9 GB | ~31 tok/s | Dense | — | Strong coding assistant |
| Qwen3-Coder-30B-A3B | 30B total | 17.0 GB | ~32 tok/s | **MoE** | 73.4% | Top coding pick — MoE speed trick |
| Gemma 4 26B-A4B | 26B total | 18.0 GB | ~30 tok/s | **MoE** | — | Google's MoE, multilingual |
| **Qwen3.6-27B** | 27B | 16.8 GB | ~12 tok/s | Dense | **77.2%** | Best coding benchmark in 24 GB |
| Qwen3.6-35B-A3B | 35B total | 21.0 GB | ~35 tok/s | **MoE** | 73.4% | Top general pick — fast MoE |

### What Doesn't Fit (Needs 48 GB+)

| Model | Params | Size | Why it fails |
|---|---|---|---|
| Gemma 4 31B (dense) | 31B | 17.4 GB | Fits, but ~10 tok/s — barely usable |
| Qwen2.5 32B | 32B | 19.0 GB | Tight — might work with reduced context |
| DeepSeek R1 70B | 70B | 42.0 GB | Needs ~48 GB unified memory |
| Qwen3-Coder-Next (~100B+) | 100B+ | 48.2 GB+ | Needs 64 GB+ |

> 💡 **Sweet spot on 24 GB:** Either a MoE model (Qwen3-Coder-30B-A3B, Gemma 4 26B-A4B) for speed, or Qwen3.6-27B if benchmark quality matters more than tokens/sec.

### Dense vs MoE Trade-off

| | Dense | MoE |
|---|---|---|
| Inference speed | Bandwidth ÷ total size | Bandwidth ÷ active params |
| RAM footprint | Proportional to params | Same as equivalent dense |
| Quality/param | Higher benchmark scores | Slightly lower per total param |
| Best for | Highest quality at a given size | Speed without losing capability |

---

## Speed Reference

Estimated performance on M4 Pro 24 GB at Q4_K_M using MLX backend:

| Model | Speed | Fits? |
|---|---|---|
| Llama 3.2 3B | ~90 tok/s | ✅ Easily |
| Phi-4 Mini 3.8B | ~80 tok/s | ✅ Easily |
| Phi-4 14B | ~33 tok/s | ✅ Comfortably |
| Qwen2.5-Coder 14B | ~31 tok/s | ✅ Comfortably |
| Qwen3-Coder-30B-A3B (MoE) | ~32 tok/s | ✅ (17 GB) |
| Gemma 4 26B-A4B (MoE) | ~30 tok/s | ✅ (18 GB) |
| **Qwen3.6-27B (dense)** | ~12 tok/s | ✅ (16.8 GB) |
| Qwen3.6-35B-A3B (MoE) | ~35 tok/s | ⚠️ (21 GB, tight) |
| Gemma 4 31B (dense) | ~10 tok/s | ⚠️ Barely usable |
| Qwen2.5 32B | ~14 tok/s | ⚠️ Context limited |
| DeepSeek R1 70B | 0 | ❌ 42 GB required |

> At ~12 tok/s, Qwen3.6-27B is still usable for code review and document analysis where you're reading, not watching the stream. Below ~8 tok/s becomes frustrating for interactive use.

---

## Setup — LM Studio & Inference Tools

### LM Studio (Recommended)

LM Studio provides a GUI for downloading, managing, and running models locally. As of v0.4.0 it supports parallel requests and an OpenAI-compatible local server.

**Install:**

```bash
# Download from https://lmstudio.ai
# Or via Homebrew:
brew install --cask lm-studio
```

**Key settings for M4 Pro:**
- Backend: **MLX** (not llama.cpp) — 20–50% faster on Apple Silicon
- GPU Layers: **All** (MLX handles this automatically)
- Context Length: 4096–8192 (adjust based on model)
- Local server: `localhost:1234` (OpenAI-compatible)

### MLX-LM (Fastest, Python)

Apple's native ML framework. Best raw inference speed on Apple Silicon.

```bash
pip install mlx-lm

# Run a model directly
mlx_lm.generate \
  --model mlx-community/Qwen3.6-27B-4bit \
  --prompt "Explain quantization in LLMs" \
  --max-tokens 500

# Start an OpenAI-compatible server
mlx_lm.server \
  --model mlx-community/Qwen3-Coder-30B-A3B-4bit \
  --port 8080
```

### Ollama

```bash
brew install ollama

# Start service
ollama serve

# Pull and run models
ollama pull qwen2.5-coder:14b
ollama run qwen2.5-coder:14b

# OpenAI-compatible API at localhost:11434
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-coder:14b","messages":[{"role":"user","content":"Hello"}]}'
```

### Backend Comparison

| Tool | Speed on Apple Silicon | Install | API |
|---|---|---|---|
| MLX-LM | ⭐ Fastest (native Metal) | pip | Yes (llmster server) |
| LM Studio (MLX) | ⭐ Fastest (uses MLX) | GUI app | Yes (port 1234) |
| Ollama | Good | brew | Yes (port 11434) |
| llama.cpp | Slower than MLX on Apple Silicon | brew/build | Yes |

---

## Monitoring Memory & Performance

### Stats (Menu Bar)

Free, visual, shows RAM/GPU/CPU in macOS menu bar.

```bash
brew install --cask stats
```

Open Stats preferences → enable RAM, CPU, GPU modules. Watch for **memory pressure** colour:
- 🟢 Green = healthy
- 🟡 Yellow = compressing (model near limit)
- 🔴 Red = swapping to disk → model will slow dramatically

### Asitop (Terminal — Apple Silicon Specific)

Shows CPU, GPU, ANE utilization and memory bandwidth live.

```bash
pip3 install asitop
sudo asitop
```

### Memory Pressure Rule

When running models near the 20 GB limit, macOS will start compressing memory. You'll see token generation speed drop suddenly. Either:
1. Reduce context window size in your inference tool
2. Switch to a smaller quantization (Q4_K_M → Q2_K)
3. Run a smaller model

---

## Hardware Upgrade Analysis — M4 Pro vs M5 Max

As of March 2026, the M5 Max is the current top-end MacBook Pro chip.

| Spec | M4 Pro (current) | M5 Max |
|---|---|---|
| Unified Memory | 24 GB | 128 GB |
| Memory Bandwidth | 273 GB/s | 614 GB/s |
| GPU Cores | 16 | 40 |
| Effective LLM Budget | ~20 GB | ~120 GB |
| Price (MacBook Pro) | ~$2,499 (paid) | ~$7,349+ |
| **Additional cost** | — | **~$4,850** |

### What M5 Max Unlocks

With ~120 GB effective RAM:

| Model | M4 Pro | M5 Max |
|---|---|---|
| Qwen3.6-27B (Q4) | ✅ 12 tok/s | ✅ ~27 tok/s |
| DeepSeek R1 70B (Q4) | ❌ Doesn't fit | ✅ ~14 tok/s |
| Llama 3.1 405B (Q2) | ❌ | ✅ ~6 tok/s |
| 30B MoE (Q4) | ✅ 32 tok/s | ✅ ~70 tok/s |
| QLoRA fine-tune 13B | ✅ | ✅ |
| QLoRA fine-tune 30B | ❌ | ✅ |
| QLoRA fine-tune 70B | ❌ | ✅ |

### The Case Against Upgrading Now

**~$4,850 additional out-of-pocket buys:**

- ~2,400–3,600 hours of cloud A100 compute (RunPod/Lambda Labs at $1.20–2.00/hr)
- You could train and experiment for **years** at cloud scale before spending that on hardware

**What you're missing that actually matters today:**
- 70B models locally (DeepSeek R1, Llama 3.1 70B): accessible via API for fractions of a cent per query
- Fine-tuning 70B+ locally: very few practitioners do this — cloud is the standard for serious fine-tuning

**Upgrade makes sense if:**
- You run large models continuously (8+ hours/day) — breakeven vs cloud compute at ~$0.50/hr
- You need full offline privacy (no API calls, no cloud)
- You are working at a lab with budget approval
- Inference speed on 30B+ models meaningfully affects your iteration speed daily

**Skip the upgrade if:**
- You're still learning and experimenting
- You work mostly with 7B–14B models (M4 Pro handles these excellently)
- Cloud compute covers your heavy lifting at lower cost

---

## eGPU — Why External NVIDIA GPUs Don't Work on Mac

> If you're considering buying an NVIDIA GPU to supplement your Mac — this path is fundamentally blocked, not just inconvenient.

**Reason 1: Apple removed eGPU support.** macOS Sonoma (14.0, Sept 2023) removed eGPU support entirely. Even AMD eGPUs that previously worked no longer function. There is no driver path, no workaround.

**Reason 2: CUDA doesn't exist on macOS.** NVIDIA's ML compute stack (CUDA → cuDNN → PyTorch CUDA) has never shipped on macOS. PyTorch on macOS routes through Apple's MPS (Metal Performance Shaders) backend, which only talks to the integrated GPU.

So connecting an RTX 4090 via Thunderbolt produces **zero ML compute** — not slow ML compute, but none at all.

### Realistic Alternatives to "eGPU on Mac"

| Option | Cost | CUDA/NVIDIA? | Tradeoff |
|---|---|---|---|
| Cloud compute (RunPod, Lambda, Vast.ai) | ~$1–3/hr for A100 | ✅ Full CUDA | Pay-per-use, best for experimentation |
| Linux desktop with RTX 4090 | ~$2,000–3,500 | ✅ Full CUDA | Dedicated machine, not supplementing Mac |
| Mac Studio M3 Ultra | ~$3,999 | ❌ Metal only | 192 GB unified memory, great local inference |
| M5 Max MacBook Pro | ~$7,349 | ❌ Metal only | 128 GB, portable, fast |

**For where most people are:** cloud compute on demand is almost always the right answer. You get NVIDIA when you need it, pay nothing when you don't.

---

## Verdict & Action Plan

### Current Machine (M4 Pro 24 GB) — What to Run

| Priority | Model | Why |
|---|---|---|
| 1. Coding | Qwen3-Coder-30B-A3B | Best coding throughput on your hardware. MoE speed trick. |
| 2. General | Qwen3.6-35B-A3B | MoE, fast, fits in 21 GB |
| 3. Benchmarks | Qwen3.6-27B | 77.2% SWE-bench — best quality in your VRAM budget |
| 4. Fast answers | Phi-4 14B or Qwen2.5-Coder 14B | ~31–33 tok/s, good quality |
| 5. Ultrafast | Llama 3.2 3B | ~90 tok/s for quick lookups |

### Four Action Items

1. **Install LM Studio or MLX-LM** — get inference running today
2. **Install Stats and asitop** — understand your actual memory usage under load
3. **Try Qwen3-Coder-30B-A3B first** — the MoE speed behaviour will surprise you
4. **Don't buy hardware yet** — exhaust what's possible on 24 GB first, then revisit the upgrade decision with real data from your own usage patterns

---

*Verified against: [LM Studio Docs](https://lmstudio.ai/docs), [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm), [Ollama.com](https://ollama.com), [Qwen3.6-27B release (April 22 2026)](https://huggingface.co/Qwen/Qwen3.6-27B), [M5 Max specs — Apple (March 2026)](https://www.apple.com/macbook-pro/) — April 2026*
