# Form Filling App

Immigration case management and USCIS form filling with a Claude agent (Modal CLI + minimal web UI).

## Summary

- Standalone CMS form-filling app (Modal CLI + minimal web UI)
- LiteParse v2 (native Python; no Node CLI in Modal images)
- Sensitive/large CMS assets excluded from git (case zip, GT JSON, GT PDFs)

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- [Modal](https://modal.com) account
- Modal secret **`ant-key`** with `ANTHROPIC_API_KEY`

## Setup

```bash
git clone https://github.com/345ishaan/form-filling-app.git
cd form-filling-app
uv sync
npm install && cd apps/web && npm install && cd ../..

# Local sample data (not in repo)
cp /path/to/sample_case.zip tasks/cms/
cp /path/to/form_*.json tasks/cms/

modal profile activate <profile>
modal secret create ant-key ANTHROPIC_API_KEY=<your-key>
```

After changing Python deps:

```bash
npm run sync:requirements
```

## Test: unit

```bash
cd apps/api
uv run pytest tests/test_cms_post_run.py -v
```

## Test: CLI (Modal)

```bash
cd apps/api
uv run python -m modal run modal_cms_app.py \
  --case-zip ../../tasks/cms/sample_case.zip \
  --form-pdf ../../tasks/cms/blank/i-140.pdf \
  --gt-json ../../tasks/cms/form_i140.json
```

Or from repo root: `npm run sim:cms`

Artifacts: Modal volume `form-filling-app-workspaces` → `runs/cms/<run_id>/`

## Test: web UI

```bash
# terminal 1
cd apps/api && uv run python -m modal serve modal_app.py
# copy Modal URL → apps/web/.env as VITE_API_BASE=https://<url>.modal.run

# terminal 2
cd apps/web && npm run dev
# open http://localhost:5173 — upload case .zip + form .pdf, chat with agent
```

Or from repo root: `npm run dev` (Modal + web together)

## What's intentionally not in git

- `tasks/cms/sample_case.zip`, `form_*.json`, `*_gt.pdf` — see [SECURITY.md](SECURITY.md)

## Layout

| Path | Purpose |
|------|---------|
| `apps/api/` | FastAPI, CMS env, Modal apps |
| `apps/web/` | Minimal React UI |
| `tasks/cms/` | Blank PDFs only in git; case zip + GT JSON local |
