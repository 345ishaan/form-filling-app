# Case and form assets

**Do not commit case documents or ground-truth filled values.** Keep them local only.

| File | In git? | Purpose |
|------|---------|---------|
| `blank/*.pdf` | Yes | Blank fillable form templates (samples) |
| `sample_case.zip` | **No** | Case document archive (~142 MB; may contain PII) |
| `form_*.json` | **No** | Ground-truth field values for offline scoring |
| `form_*_gt.pdf` | **No** | Reference filled PDFs for local eval only |

## Local setup (after clone)

```bash
cp /path/to/sample_case.zip tasks/cms/
cp /path/to/form_*.json tasks/cms/    # optional
```

Use any case `.zip` the agent can search; use any blank AcroForm PDF under `blank/` or upload via the UI.

## CLI example

```bash
cd apps/api
uv run python -m modal run modal_cms_app.py \
  --case-zip ../../tasks/cms/sample_case.zip \
  --form-pdf ../../tasks/cms/blank/i-140.pdf \
  --gt-json ../../tasks/cms/form_i140.json
```

Ground-truth JSON is **scoring-only** — never sent to the agent sandbox.
