# cowork-artifacts

Personal technical guides, built from Markdown and published via GitHub Pages.

**Live site:** https://ten-jampa.github.io/cowork-artifacts

## Guides

| Guide | Description |
|---|---|
| [Linux GPU Setup](guides/linux-gpu-guide.md) | NVIDIA/AMD driver install, CUDA, ROCm, PyTorch, LLM inference — 2026 edition |
| [Local LLMs on Apple Silicon](guides/local-llm-guide.md) | M4 Pro 24 GB — model landscape, speed, hardware upgrade analysis |

## How it works

```
guides/        ← edit these (.md source files)
generated/     ← build output, do not edit directly
scripts/
  build.py     ← Markdown → HTML generator
```

Edit a `.md` file in `guides/`, push to `main` — GitHub Actions runs `build.py`
and deploys the output to GitHub Pages automatically.

## Local development

```bash
# Install dependencies (one time)
pip install markdown pymdown-extensions pygments jinja2

# Build all guides
python3 scripts/build.py

# Build a single guide
python3 scripts/build.py --file linux-gpu-guide.md

# Watch mode (auto-rebuild on save)
pip install watchdog
python3 scripts/build.py --watch
```
