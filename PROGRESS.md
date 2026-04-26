# PROGRESS — Design Decisions & Historical Context

A log of architectural decisions, what was tried, what changed, and why.

---

## April 26, 2026 — Initial build

### Context
Starting point: two standalone HTML files produced during a Cowork session covering local LLM setup on an M4 Pro MacBook and Linux GPU configuration.

**Files:**
- `local-llm-guide.html` — M4 Pro model landscape, LM Studio, upgrade analysis
- `linux-gpu-guide.html` — NVIDIA/AMD drivers, CUDA 13.x, ROCm 7.x, PyTorch, LLM inference

---

## Decision: Flat HTML → Markdown as source of truth

**What was tried first:** Keeping flat HTML files as the primary artifact and discussing building a post-processing MCP on top of them.

**Why it was rejected:** HTML is a presentation format, not a knowledge format. Maintaining content inside `<div class="code-block">` tags is error-prone and produces unreadable git diffs. An MCP built on top of HTML would spend most of its effort fighting the format — stripping tags, extracting code blocks, inferring structure. The wrong abstraction layer.

**Decision:** Convert to Markdown as source of truth, generate HTML from that. Markdown is readable by humans, LLMs, Git, Obsidian, GitHub, and every static site generator. If you need a pretty interactive guide, you generate it from MD. If you want Claude to query it, you point the MCP at MD.

**Folder structure adopted:**
```
docs/          ← source of truth (edit these)
generated/     ← build output
scripts/
  build.py     ← Markdown → HTML generator
```

---

## Decision: Custom build.py → MkDocs Material

**What was built first:** A custom Python `build.py` using `python-markdown` + `pymdown-extensions` + `jinja2`. It read `.md` files, applied a Jinja2 template with a Claude dark theme, injected copy buttons, extracted H2s for nav, and output to `generated/`.

**Why it was good enough to ship:**
- Correct separation of source and output
- CI/CD via GitHub Actions
- Build artifacts not committed to `main`

**Why it was replaced:**

Rolling a custom static site generator is an anti-pattern for a documentation workflow. The custom script had real weaknesses:
- No search
- Nav was fragile (scraping H2s, no hierarchy)
- Jinja2 template embedded in a Python string — painful to modify
- No incremental builds
- No link checking
- No dark/light toggle
- No mobile-optimised sidebar

**Decision:** Replace with **MkDocs Material** — the genuine best practice for Python developers writing technical docs in 2026. Used at Google, Stripe, FastAPI, Anthropic. Gives out of the box: full-text search, dark/light mode toggle, sidebar nav, TOC with scroll tracking, copy buttons, admonition blocks, tabbed content, mobile drawer, previous/next navigation.

The `.md` files required zero changes — only the build tooling changed.

---

## Decision: Light design

**Original:** Claude dark theme (GitHub dark palette — `#0d1117` background).

**Changed to:** Light theme on user request.

For MkDocs Material this is a single `scheme: default` flag in `mkdocs.yml`, and the dark/light toggle is now built in — users can switch themselves.

---

## Decision: GitHub Actions deployment

**Approach:** `mkdocs gh-deploy --force` triggered on push to `main` when any file in `docs/` or `mkdocs.yml` changes.

- `main` branch: source only (`.md`, `mkdocs.yml`, workflow)
- `gh-pages` branch: built output, managed entirely by CI
- No build artifacts committed to `main`

**GitHub Pages settings:** Source = `gh-pages` branch, folder = `/ (root)`.

**Live URL:** `https://ten-jampa.github.io/cowork-artifacts`

---

## Folder structure (current)

```
cowork-artifacts/
  docs/
    index.md                   ← site landing page
    linux-gpu-guide.md         ← Linux GPU setup (source)
    local-llm-guide.md         ← Apple Silicon local LLMs (source)
  .github/
    workflows/
      build-and-deploy.yml     ← CI: build + deploy on push
  mkdocs.yml                   ← MkDocs Material config
  README.md                    ← repo homepage
  .gitignore                   ← excludes site/, .DS_Store
  PROGRESS.md                  ← this file
```

---

## What was discarded

| Artifact | Why removed |
|---|---|
| `local-llm-guide.html` (root) | Superseded by `docs/local-llm-guide.md` + MkDocs |
| `linux-gpu-guide.html` (root) | Superseded by `docs/linux-gpu-guide.md` + MkDocs |
| `scripts/build.py` | Replaced by MkDocs — a purpose-built, maintained tool |
| `generated/` folder | Replaced by MkDocs `site/` output (gitignored) |
| `guides/` folder | Renamed to `docs/` to match MkDocs convention |

---

## Future decisions to make

- **MCP layer:** If we build an MCP to query/update these guides, it should operate on the `.md` files in `docs/`, not on generated HTML.
- **Content updates:** When CUDA/ROCm versions change, edit the relevant `.md`, push — CI rebuilds and redeploys automatically.
- **New guides:** Add a new `.md` to `docs/`, add a line to the `nav:` section in `mkdocs.yml`, push.
