# Linux GPU Setup — Start to Finish (2026 Edition)

> **Verified:** CUDA 13.2 · ROCm 7.1.1 · Ubuntu 24.04 LTS · PyTorch 2.9 — April 2026

A comprehensive guide for getting your NVIDIA or AMD GPU fully operational on Linux for AI/ML inference and training.

---

## Table of Contents

1. [Pre-flight Checklist](#pre-flight-checklist)
2. [Choosing Your Distro](#choosing-your-distro)
3. [GPU Buying Guide](#gpu-buying-guide)
4. [Secure Boot](#secure-boot)
5. [Driver Installation](#driver-installation)
6. [CUDA Toolkit & ROCm SDK](#cuda-toolkit-rocm-sdk)
7. [Laptop Hybrid Graphics (Optimus)](#laptop-hybrid-graphics-optimus)
8. [Python Environment Setup](#python-environment-setup)
9. [PyTorch — GPU Installation](#pytorch-gpu-installation)
10. [LLM Inference Frameworks](#llm-inference-frameworks)
11. [GPU Monitoring Tools](#gpu-monitoring-tools)
12. [Troubleshooting](#troubleshooting)

---

## Pre-flight Checklist

Run these commands to understand exactly what you're working with before touching a single driver.

### Identify Your GPU

```bash
lspci -nn | grep -i "VGA\|3D\|Display"

# Detailed GPU info
lspci -v -s $(lspci | grep -i vga | cut -d' ' -f1)
```

### Check Kernel & OS

```bash
uname -r           # kernel version
lsb_release -a     # distro/version
cat /proc/driver/nvidia/version  # existing NVIDIA driver?
```

### Check RAM & Secure Boot

```bash
free -h              # system RAM
df -h /              # disk space
mokutil --sb-state   # Secure Boot on/off?
lscpu | grep -i arch # x86_64 or ARM?
```

### Check for Existing Drivers

```bash
nvidia-smi                                     # NVIDIA loaded?
rocm-smi                                       # AMD ROCm loaded?
lsmod | grep -i "nvidia\|nouveau\|amdgpu"
```

### What Your Output Means

| You see | What it means | Next step |
|---|---|---|
| `NVIDIA Corporation` in lspci | NVIDIA discrete GPU | Follow NVIDIA driver path |
| `AMD/ATI` in lspci | AMD discrete GPU | Follow AMD/ROCm path |
| Two GPUs (Intel + NVIDIA) | Hybrid Optimus laptop | Read Optimus section first |
| `SecureBoot enabled` | Must sign kernel modules | Read Secure Boot section |
| `nouveau` in lsmod | Open-source driver loaded — must blacklist | See blacklist step |

---

## Choosing Your Distro

### Ubuntu 24.04 LTS — Best Choice

The de-facto standard for ML/AI in 2026. Largest community, most tutorials, official NVIDIA support, and `ubuntu-drivers` makes driver install nearly foolproof. LTS until 2029.

- **Kernel:** 6.8 (HWE for 6.11+)
- **CUDA:** Full support
- **ROCm:** Full support

### Fedora 41/42 — Cutting Edge

Best if you want the latest PyTorch and kernel versions without waiting for Ubuntu backports. NVIDIA drivers restored in GNOME Software with Secure Boot support.

- **Driver install:** RPM Fusion repo
- **Good for:** experimenting, latest libraries

### Pop!_OS 24.04 — Plug-and-Play

Ships NVIDIA drivers pre-installed. Best out-of-the-box GPU experience. Based on Ubuntu so all the same tools work.

- **NVIDIA ISO:** driver bundled
- **Optimus:** handled automatically

> ⚠️ **Avoid** Arch, Gentoo, or anything bleeding-edge for your first Linux GPU setup. Driver/kernel mismatches are the #1 cause of broken installs.

> 💡 This guide primarily targets **Ubuntu 24.04 LTS** — commands are noted where they differ on Fedora.

---

## GPU Buying Guide

Before buying anything, three questions determine everything: **How much VRAM? NVIDIA or AMD? New or used?**

> Street prices are April 2026. MSRP and street pricing are very different due to Blackwell (RTX 50xx) supply constraints.

### Step 1 — VRAM Sizing

VRAM is the single most important spec for LLM inference. A lower-generation GPU with more VRAM will almost always outperform a newer GPU with less for running large models. Model doesn't fit in VRAM → spills to system RAM → 10–100× slower.

| VRAM | What fits (Q4_K_M quantized) | Use case | Min GPU |
|---|---|---|---|
| 8 GB | Up to ~7B params | Fast local assistant, coding help | RTX 3070 / 4060 Ti |
| 12 GB | Up to ~13B params | Good quality chat, coding, summarization | RTX 3080 12GB / 4070 |
| 16 GB | Up to ~20B params + MoE 30B | Strong local models, QLoRA fine-tuning ≤13B | RTX 5070 Ti / 4070 Ti Super |
| 24 GB | Up to ~33B dense, or 70B Q2 | Serious inference, fine-tuning up to 20B | RTX 3090 / 4090 / 7900 XTX |
| 32 GB | Up to ~50B Q4, or 70B Q2–Q3 | Research, large model experimentation | RTX 5090 |
| 48+ GB | 70B Q4, 100B+ Q4 | Production serving, full fine-tuning 30B+ | RTX A6000 / multi-GPU |

### Step 2 — NVIDIA vs AMD

**NVIDIA — Clear winner for ML/AI**

- CUDA ecosystem: PyTorch, llama.cpp, vLLM, TensorRT, bitsandbytes — all primary NVIDIA targets
- QLoRA fine-tuning (bitsandbytes) requires NVIDIA — won't run on AMD without patches
- Real-world throughput advantages over AMD are 20–40% beyond raw bandwidth

**AMD — Legit with trade-offs**

- ROCm 7.x in 2026 is genuinely usable. PyTorch 2.9 has official ROCm wheels
- RX 7900 XTX: 24 GB for ~$700–850 — strong value for inference-only
- RX 9070 XT: official ROCm 7.2 support, 16 GB, ~$600
- bitsandbytes (4-bit QLoRA) is NVIDIA-only. Flash-attention needs manual ROCm ports
- **Verdict:** Good for inference-only. Avoid for fine-tuning

### Step 3 — Specific GPU Recommendations

#### Budget Tier (~$500–800)

| GPU | VRAM | Bandwidth | Street Price | Verdict |
|---|---|---|---|---|
| RTX 3090 (used) | 24 GB GDDR6X | 936 GB/s | ~$550–700 | ⭐ Best VRAM/$ |
| RX 9070 XT (AMD) | 16 GB GDDR6 | 896 GB/s | ~$599–650 | Official ROCm 7.2 |
| RX 7900 XTX (AMD) | 24 GB GDDR6 | 960 GB/s | ~$700–850 | 24 GB for less |

**Best pick:** The used RTX 3090 is the VRAM king at this price — 24 GB for ~$600. Only downsides: large 3-slot cooler, runs hot (350W), buying used.

#### Sweet Spot Tier (~$800–1,400)

| GPU | VRAM | Bandwidth | Street Price | Verdict |
|---|---|---|---|---|
| RTX 4070 Ti Super | 16 GB GDDR6X | 672 GB/s | ~$799–900 | Best efficiency |
| RTX 4090 | 24 GB GDDR6X | 1,008 GB/s | ~$1,200–1,600 | Best Ada value |
| RTX 5070 Ti | 16 GB GDDR7 | 896 GB/s | ~$1,000–1,200 | MSRP $749, street ~$1,000 |
| RTX 5080 | 16 GB GDDR7 | 960 GB/s | ~$1,200–1,600 | 16 GB at high price |

**Best pick:** RTX 4090 at ~$1,200–1,400 (post-5090 price drop) — 24 GB, best Ada Lovelace tensor performance.

#### Enthusiast Tier ($1,500+)

| GPU | VRAM | Bandwidth | Street Price | Verdict |
|---|---|---|---|---|
| RTX 5090 | 32 GB GDDR7 | 1,792 GB/s | ~$3,000–4,000 | Best consumer GPU |
| RTX A6000 (used) | 48 GB GDDR6 | 768 GB/s | ~$2,500–3,500 | Workstation VRAM king |

**RTX 5090:** 1,792 GB/s bandwidth (78% more than 4090). MSRP $1,999 but street ~$3,800 due to supply. At MSRP it's exceptional; at street price, cloud GPU time is a serious competitor.

> ⚠️ **Avoid the RTX 5080 and 5070 Ti for AI.** Only 16 GB GDDR7 at $1,000–1,600 street, when a 24 GB RTX 4090 costs the same. NVIDIA's 16 GB decision on these cards is widely criticised by the ML community.

### Quick Decision Matrix

| Your situation | Get this | Why |
|---|---|---|
| Budget ~$600, want most VRAM | Used RTX 3090 | 24 GB for ~$600, full CUDA |
| Budget ~$600, want AMD + official ROCm | RX 9070 XT | Official ROCm 7.2, 16 GB GDDR6 |
| Budget ~$800, best efficiency | RTX 4070 Ti Super | Ada, efficient, 16 GB, full CUDA |
| Best value for serious ML | RTX 4090 | 24 GB, best Ada tensor cores |
| Max VRAM, cost no object | RTX 5090 | 32 GB GDDR7, 1,792 GB/s |
| Want 40B+ without $3,000+ | Used RTX A6000 | 48 GB workstation-grade |
| Inference-only, tight budget, okay with AMD friction | RX 7900 XTX | 24 GB for ~$700–850 |
| Primarily fine-tuning (QLoRA) | RTX 4090 or 3090 | bitsandbytes requires NVIDIA; 24 GB comfortable |

---

## Secure Boot

UEFI Secure Boot prevents unsigned kernel modules from loading. NVIDIA's proprietary driver is a kernel module, so this must be resolved first.

### Option A — Disable Secure Boot (Simplest)

Reboot → enter UEFI/BIOS (F2, F12, or Del) → Security → Secure Boot → **Disabled**. 90% of ML developers do this.

### Option B — MOK Key Enrollment

Generate a Machine Owner Key pair, sign the NVIDIA kernel module, enroll the public cert in UEFI's MOK database.

```bash
# Create directory for keys
sudo mkdir -p /var/lib/shim-signed/mok
cd /var/lib/shim-signed/mok

# Generate 2048-bit RSA keypair (valid 10 years)
sudo openssl req -new -x509 -newkey rsa:2048 \
    -keyout MOK.priv -out MOK.pem \
    -days 3650 -subj "/CN=MOK Signing Key/" \
    -nodes

# Convert to DER format (required by mokutil)
sudo openssl x509 -in MOK.pem -out MOK.der -outform DER

# Enroll public cert (will prompt for a one-time password)
sudo mokutil --import MOK.der

# Reboot — UEFI MokManager will appear, choose "Enroll MOK"
sudo reboot

# After reboot: verify key enrolled
mokutil --list-enrolled | grep -A2 "MOK Signing"
```

```bash
# Sign each NVIDIA kernel module (run after driver install)
for module in nvidia nvidia_modeset nvidia_uvm nvidia_drm; do
    sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file \
      sha256 \
      /var/lib/shim-signed/mok/MOK.priv \
      /var/lib/shim-signed/mok/MOK.pem \
      $(modinfo -n $module)
done

# Verify signing
modinfo nvidia | grep "signer"
```

> ⚠️ **Known 2025 issue:** Some kernels compress modules as `.ko.zst`. The sign script expects `.ko`. If you get "file not found" errors: `sudo zstd -d module.ko.zst`, sign, re-compress.

---

## Driver Installation

> 🚫 **Rule #1:** Pick exactly ONE installation method. Mixing methods (apt + runfile, or PPA + official repo) will break your system.

### NVIDIA Drivers

#### Method A — ubuntu-drivers (Recommended)

```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install kernel headers
sudo apt install -y linux-headers-$(uname -r)

# See what ubuntu-drivers recommends
ubuntu-drivers devices

# Auto-install recommended driver
sudo ubuntu-drivers install

# OR install specific version (e.g., 570)
sudo apt install -y nvidia-driver-570

# Reboot — mandatory
sudo reboot
```

#### Method B — Official NVIDIA Network Repository

Use when you need a very specific driver version or ubuntu-drivers doesn't find your GPU (new hardware).

```bash
# Add NVIDIA's keyring and repository
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install driver (also pulls CUDA)
sudo apt install -y cuda-drivers

# OR driver-only (open-source kernel modules)
sudo apt install -y nvidia-open-570

sudo reboot
```

> 💡 As of driver 515+, NVIDIA ships **open-source kernel modules** (`nvidia-open`) alongside proprietary ones. Preferred on Turing/RTX 2000+ architecture. Not the same as `nouveau`.

#### Blacklist Nouveau

Only necessary if `ubuntu-drivers` didn't do it automatically.

```bash
# Check if nouveau is loaded
lsmod | grep nouveau

# If it appears, blacklist it
echo -e "blacklist nouveau\noptions nouveau modeset=0" \
    | sudo tee /etc/modprobe.d/blacklist-nouveau.conf

sudo update-initramfs -u
sudo reboot

# Verify nouveau is gone (should return nothing)
lsmod | grep nouveau
```

#### Verify NVIDIA Driver

```bash
nvidia-smi
# Expected: shows driver version, CUDA Version, GPU name and VRAM usage

cat /proc/driver/nvidia/version
```

#### Fedora — RPM Fusion Method

```bash
# Enable RPM Fusion
sudo dnf install -y \
    https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
    https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Install NVIDIA driver
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda

# Wait for kernel module build (~5 min)
sudo akmods --force
sudo reboot
```

### AMD Drivers

The `amdgpu` open-source driver is built into the Linux kernel — your GPU will display fine without ROCm. ROCm is only needed for ML compute.

```bash
# amdgpu should load by default on GCN 3.0+ GPUs
lsmod | grep amdgpu
dmesg | grep -i amdgpu | head -20
```

#### ROCm Installation (Ubuntu 24.04)

```bash
# Install prerequisites
sudo apt update
sudo apt install -y wget gpg

# Add AMD ROCm repo key
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor \
    | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Add ROCm 7.x repository
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
    https://repo.radeon.com/rocm/apt/7.1.1 noble main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update

# Install ROCm compute stack
sudo apt install -y rocm-hip-sdk rocm-dkms

# Add your user to render and video groups
sudo usermod -aG render,video $USER

sudo reboot
```

#### Verify ROCm

```bash
rocm-smi         # GPU status (like nvidia-smi for AMD)
rocminfo         # detailed hardware info
ls /dev/kfd /dev/dri   # both must exist
```

> ⚠️ **AMD GPU Compatibility:** ROCm officially supports RX 6000 (RDNA 2), RX 7000 (RDNA 3), RX 9000 (RDNA 4), and MI series. Older GPUs have community support only. Always verify your GPU on the [official compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).

---

## CUDA Toolkit & ROCm SDK

### NVIDIA — CUDA 13.x

> 💡 Modern PyTorch (2.6+) bundles its own CUDA runtime inside the pip wheel. You only need the NVIDIA **driver** for PyTorch inference. Install the full CUDA Toolkit only if you plan to compile custom CUDA kernels, use `nvcc`, or build llama.cpp from source.

```bash
# Install CUDA Toolkit 13.2 (assumes NVIDIA repo added in driver step)
sudo apt install -y cuda-toolkit-13-2
```

#### Add CUDA to PATH

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

```bash
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi | grep "CUDA Version"
```

#### CUDA Version Compatibility

| CUDA Version | Min Driver (Linux) | PyTorch Wheel | Status |
|---|---|---|---|
| 13.2 | 570.00+ | cu132 | ✅ Current |
| 12.9 | 550.00+ | cu129 | ✅ Stable |
| 12.4 | 525.85+ | cu124 | ⚠️ Aging |
| 12.1 | 525.85+ | cu121 | ⚠️ Legacy |
| 11.8 | 450.80+ | cu118 | ❌ EOL |

> Rule: CUDA version in the PyTorch wheel must be ≤ your driver's max CUDA version (shown top-right in `nvidia-smi`).

### AMD — ROCm 7.x

```bash
# Check installed ROCm version
cat /opt/rocm/lib/rocm_version
hipcc --version    # HIP compiler (ROCm's CUDA equivalent)

# Check your GPU architecture target
rocminfo | grep -i "gfx"
# gfx1100 = RX 7900 XTX (RDNA 3)
# gfx1030 = RX 6800 XT (RDNA 2)
# gfx942  = MI300X (data center)
# gfx1201 = RX 9070 XT (RDNA 4)
```

#### Docker — Fastest ROCm Onramp

```bash
# Install Docker
sudo apt install -y docker.io
sudo usermod -aG docker $USER

# Pull official AMD ROCm + PyTorch image
docker pull rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.11_pytorch_2.9

# Run with GPU access
docker run -it --device=/dev/kfd --device=/dev/dri \
    --group-add render --group-add video \
    -v $(pwd):/workspace \
    rocm/pytorch:rocm7.1.1_ubuntu22.04_py3.11_pytorch_2.9 bash

# Inside container — verify
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Laptop Hybrid Graphics (Optimus)

If your laptop has both an Intel/AMD iGPU and a discrete NVIDIA GPU, you have Optimus. Symptom: `nvidia-smi` works but Python can't see the GPU.

### Optimus Modes

| Mode | What runs on NVIDIA | Power | Best for |
|---|---|---|---|
| Hybrid (default) | Apps launched with `prime-run` | Low | Battery life + occasional GPU use |
| NVIDIA only | Everything | High | Sustained ML training sessions |
| iGPU only | Nothing | Very low | Travel, no ML needed |

### prime-select (Ubuntu/Debian)

```bash
# Install nvidia-prime
sudo apt install -y nvidia-prime

# Check current mode
prime-select query

# Switch to NVIDIA-only (needs reboot)
sudo prime-select nvidia
sudo reboot

# Switch back to hybrid
sudo prime-select on-demand
sudo reboot

# In on-demand mode: prefix commands with prime-run
prime-run python3 -c "import torch; print(torch.cuda.device_count())"
```

### CUDA_VISIBLE_DEVICES — Per-command GPU Control

```bash
# Force specific GPU for a single command
CUDA_VISIBLE_DEVICES=0 python3 my_script.py

# Use all GPUs
CUDA_VISIBLE_DEVICES=0,1 python3 multi_gpu_script.py

# CPU-only mode
CUDA_VISIBLE_DEVICES="" python3 my_script.py
```

> ⚠️ **Optimus + HDMI/DisplayPort:** On many laptops, external display ports are wired to the iGPU. In NVIDIA-only mode you may lose your external display. Hybrid mode is usually better for a docked ML workstation setup.

---

## Python Environment Setup

Never install ML packages into the system Python. Always use isolated environments.

### Mamba (Recommended for GPU/ML)

Mamba solves environments in seconds vs conda's 3–4 minutes. Better for GPU stacks with native dependencies.

```bash
# Download Miniforge (conda + mamba, free license)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# Init shell integration
~/miniforge3/bin/conda init bash   # or zsh, fish
source ~/.bashrc

# Verify
mamba --version
```

#### Create a GPU ML Environment

```bash
# Create environment with Python 3.11
mamba create -n ml python=3.11 -y
mamba activate ml
pip install --upgrade pip
```

### venv + pip (Lightweight Alternative)

```bash
python3 -m venv ~/.venvs/ml
source ~/.venvs/ml/bin/activate

# Easy alias
echo 'alias mlenv="source ~/.venvs/ml/bin/activate"' >> ~/.bashrc
```

> 💡 Use mamba for GPU-heavy or LLM-related work. Use venv+pip for lightweight pure-Python projects.

---

## PyTorch — GPU Installation

> ⚠️ Since PyTorch 2.6.0 (late 2024), **conda install pytorch is discontinued**. Use pip. PyTorch wheels bundle their own CUDA runtime — you only need a compatible NVIDIA driver, not the full CUDA Toolkit.

### NVIDIA (CUDA)

```bash
# Activate your environment first
mamba activate ml

# PyTorch 2.9 + CUDA 12.9 (bundled runtime — recommended stable)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# Or CUDA 13.2 (requires driver 570+)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu132
```

> 💡 Not sure which wheel? Run `nvidia-smi` and look at "CUDA Version" top-right. Pick a wheel version ≤ that number. When in doubt, `cu129` is safest.

### AMD (ROCm)

```bash
# AMD-hosted wheels (recommended by AMD)
pip install torch torchvision torchaudio \
    --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/

# Or PyTorch.org ROCm wheel
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm7.1
```

### Verify GPU

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"CUDA version:    {torch.version.cuda}")
print(f"GPU count:       {torch.cuda.device_count()}")

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU name: {gpu.name}")
    print(f"VRAM:     {gpu.total_memory / 1e9:.1f} GB")

    # Actual compute test
    a = torch.randn(4096, 4096, device='cuda')
    b = torch.randn(4096, 4096, device='cuda')
    c = torch.matmul(a, b)
    print(f"Matrix multiply OK — shape: {c.shape}")
```

### Core ML Libraries

```bash
# Hugging Face ecosystem
pip install transformers accelerate datasets huggingface_hub

# Quantization (run large models in less VRAM)
pip install bitsandbytes   # 4-bit/8-bit quantization (NVIDIA only)
pip install auto-gptq      # GPTQ quantized model support

# Fine-tuning
pip install peft trl       # LoRA/QLoRA + RLHF tooling

# Utilities
pip install numpy scipy scikit-learn matplotlib jupyterlab
```

---

## LLM Inference Frameworks

### Ollama — Easiest Setup

One-liner install, auto-detects GPU, huge model library, OpenAI-compatible API on port 11434.

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run models
ollama pull llama3.2            # 3B — fast, fits any GPU
ollama pull qwen2.5-coder:14b   # 14B — needs ~10 GB VRAM
ollama pull qwen3:30b-a3b       # 30B MoE — ~17 GB VRAM

# Run interactively
ollama run llama3.2

# Check GPU layer count
ollama run llama3.2 --verbose 2>&1 | grep "gpu layers"

# Force all layers to GPU
OLLAMA_NUM_GPU=99 ollama run qwen2.5-coder:14b

# OpenAI-compatible API
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello"}]}'
```

### llama.cpp — Most Flexible

Build with CUDA/ROCm/Vulkan, supports every GGUF model, layer offloading (split across GPU+CPU).

```bash
# Install build deps
sudo apt install -y build-essential cmake git

# Clone and build with CUDA
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# For AMD ROCm instead
cmake -B build -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Download a GGUF model
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/Qwen2.5-Coder-14B-Instruct-GGUF',
  'Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf', local_dir='./models')
"

# Run with all layers on GPU (-ngl -1 = all layers)
./build/bin/llama-cli \
  -m ./models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
  -ngl -1 -c 4096 \
  -p "Write a Python function to sort a list"

# Start as OpenAI-compatible server
./build/bin/llama-server \
  -m ./models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
  -ngl -1 --port 8080
```

### vLLM — Production Throughput

PagedAttention for high-throughput inference, continuous batching. Requires Hugging Face models (not GGUF). Minimum ~8 GB VRAM for 7B models.

```bash
# Install
pip install vllm

# Serve a model (OpenAI-compatible on port 8000)
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dtype auto \
  --max-model-len 8192 \
  --port 8000

# AWQ quantized (lower VRAM)
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
  --quantization awq \
  --dtype half
```

---

## GPU Monitoring Tools

### nvidia-smi (Built-in)

```bash
# Snapshot
nvidia-smi

# Live refresh every 0.5 seconds
watch -n 0.5 nvidia-smi

# Log to CSV
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,\
utilization.gpu,utilization.memory,memory.used,memory.free \
--format=csv -l 1 | tee gpu_log.csv
```

### nvtop — htop for GPUs (NVIDIA + AMD)

```bash
sudo apt install -y nvtop
nvtop   # F2 for setup, Q to quit
```

### nvitop — Richest NVIDIA UI

```bash
pip install nvitop
nvitop                 # full UI
nvitop -m compact      # compact (good in a tmux pane)

# Use as Python library inside training loop
from nvitop import Device
for d in Device.all():
    print(d.name(), d.memory_used_human())
```

### AMD — rocm-smi

```bash
rocm-smi                  # snapshot
watch -n 1 rocm-smi       # live
rocm-smi --showmeminfo vram
```

### tmux Split-Pane Monitor Setup

```bash
sudo apt install -y tmux

# Start session with monitor in right pane
tmux new-session -s ml \; \
  split-window -h \; \
  send-keys 'nvitop -m compact' Enter \; \
  select-pane -t 0
# Left pane = your work, right pane = GPU monitor
```

---

## Troubleshooting

### nvidia-smi: command not found / has failed

```bash
# Check if nouveau is blocking nvidia
lsmod | grep nouveau    # bad if this returns anything
lsmod | grep nvidia     # should show multiple modules

# Check kernel module status
dmesg | grep -i nvidia
journalctl -b | grep -i nvidia

# Re-blacklist and rebuild initramfs
echo -e "blacklist nouveau\noptions nouveau modeset=0" \
    | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u && sudo reboot

# Force-load driver
sudo modprobe nvidia
```

### torch.cuda.is_available() Returns False

```bash
# Step 1: Confirm driver works
nvidia-smi

# Step 2: Check driver's max CUDA version
nvidia-smi | grep "CUDA Version"   # e.g. CUDA Version: 13.2

# Step 3: Check PyTorch CUDA version
python3 -c "import torch; print(torch.version.cuda)"

# Step 4: Reinstall with correct wheel if mismatch
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# Step 5: Confirm you're in the right environment
which python3        # must NOT be /usr/bin/python3
mamba activate ml
```

### Laptop: GPU Detected but Model Runs on CPU (Optimus)

```bash
prime-select query

# Run explicitly on NVIDIA GPU
prime-run python3 my_script.py

# Or switch to NVIDIA-only
sudo prime-select nvidia && sudo reboot

# Verify GPU utilization is non-zero during inference
nvidia-smi dmon
```

### CUDA Out of Memory

```python
import torch

# Free cached memory
torch.cuda.empty_cache()

# Diagnose usage
print(torch.cuda.memory_summary())
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached:    {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Reduce memory footprint:
# 1. Mixed precision (fp16/bf16)
model = model.half()

# 2. Gradient checkpointing
model.gradient_checkpointing_enable()

# 3. 4-bit quantization (bitsandbytes, NVIDIA only)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
```

### Kernel Update Broke NVIDIA Driver

```bash
# Check DKMS status
dkms status

# Rebuild for current kernel
sudo dkms autoinstall

# Or reinstall driver
sudo apt install --reinstall nvidia-driver-570
sudo reboot

# Check build log if still failing
cat /var/lib/dkms/nvidia/570*/build/make.log | tail -50
```

### Ollama Using CPU Instead of GPU

```bash
# Check logs for GPU detection
journalctl -u ollama --no-pager | grep -i "gpu\|cuda\|rocm"

# Run in foreground to see output
ollama serve   # look for "using GPU" vs "0 GPU layers"

# Fix: reinstall AFTER driver is working
curl -fsSL https://ollama.com/install.sh | sh

# Fix: verify nvidia-smi works first
nvidia-smi
```

### Full Diagnostic Script

```bash
#!/bin/bash
echo "===== GPU DIAGNOSTIC ====="
echo "--- GPU Hardware ---"
lspci -nn | grep -i "VGA\|3D\|Display"
echo ""
echo "--- Kernel Modules ---"
lsmod | grep -E "nvidia|nouveau|amdgpu" || echo "none found"
echo ""
echo "--- NVIDIA Driver ---"
nvidia-smi 2>/dev/null || echo "nvidia-smi not found"
echo ""
echo "--- ROCm ---"
rocm-smi 2>/dev/null || echo "rocm-smi not found"
echo ""
echo "--- Python / PyTorch ---"
python3 -c "
try:
    import torch
    print(f'PyTorch {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f'VRAM: {mem/1e9:.1f} GB')
except ImportError:
    print('PyTorch not installed')
" 2>/dev/null
echo ""
echo "--- CUDA Toolkit ---"
nvcc --version 2>/dev/null || echo "nvcc not found"
echo "===== END DIAGNOSTIC ====="
```

---

*Verified against: [NVIDIA CUDA Install Guide (v13.2)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), [Ubuntu NVIDIA Driver Docs](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/), [ROCm PyTorch Install (AMD)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html), [nvtop GitHub](https://github.com/Syllo/nvtop), [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp) — April 2026*
