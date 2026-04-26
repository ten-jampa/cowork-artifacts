# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal MkDocs Material static site that publishes technical guides to GitHub Pages at `https://ten-jampa.github.io/cowork-artifacts/`. The only dependency is `mkdocs-material==9.7.6`.

## Commands

```bash
# Install (one-time)
pip install -r requirements.txt

# Local dev server with live reload at http://127.0.0.1:8000
mkdocs serve

# Build static site into site/ (not committed)
mkdocs build

# Deploy directly to GitHub Pages (CI does this automatically)
mkdocs gh-deploy --force
```

## How it works

- **Source**: all guide content lives in `docs/` as `.md` files
- **Config**: `mkdocs.yml` controls nav, theme, and extensions — update `nav:` when adding a new guide
- **Deploy**: every push to `main` triggers `.github/workflows/build-and-deploy.yml`, which runs `mkdocs gh-deploy --force` via GitHub Actions
- No build scripts, no `scripts/` directory, no `generated/` directory — the README is outdated on this point; MkDocs handles everything

## Adding a new guide

1. Create `docs/<guide-name>.md`
2. Add an entry under `nav:` in `mkdocs.yml`
3. Push to `main` — CI deploys automatically

## MkDocs Material features in use

The guides use Material-specific syntax: admonition blocks (`!!! note`, `!!! warning`, `!!! danger`), `=== "Tab"` tabbed sections, `- [x]` task lists, and fenced code blocks with language tags for syntax highlighting. Keep new content consistent with these patterns.
