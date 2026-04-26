# Technical Guides

Personal technical reference documents covering AI/ML infrastructure, local model inference, and hardware setup.

---

## Guides

### 🐧 [Linux GPU Setup](linux-gpu-guide.md)

A verified, start-to-finish guide for getting NVIDIA or AMD GPUs working on Linux for AI/ML. Covers driver installation, CUDA 13.x, ROCm 7.x, PyTorch, LLM inference frameworks (Ollama, llama.cpp, vLLM), GPU buying recommendations, and a full troubleshooting reference.

**Verified:** CUDA 13.2 · ROCm 7.1.1 · Ubuntu 24.04 LTS · PyTorch 2.9 · April 2026

---

### 🍎 [Local LLMs on Apple Silicon](local-llm-guide.md)

Running large language models locally on an M4 Pro MacBook with 24 GB unified memory. Covers what models fit, how fast they run, MoE vs dense trade-offs, LM Studio and MLX setup, and a cost/benefit analysis of upgrading to M5 Max.

**Verified:** M4 Pro 24 GB · MLX · LM Studio 0.4+ · Qwen3.6-27B · April 2026

---

*Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Source on [GitHub](https://github.com/ten-jampa/cowork-artifacts).*
