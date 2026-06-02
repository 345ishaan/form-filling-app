# CMS case data

**Do not commit case documents or ground-truth filled values.** They stay on your machine only.

| File | In git? | Purpose |
|------|---------|---------|
| `blank/*.pdf` | Yes | Public USCIS blank form templates |
| `sample_case.zip` | **No** | Case evidence zip (~142 MB, may contain PII) |
| `form_*.json` | **No** | GT filled values for offline P/R (contains PII) |
| `form_*_gt.pdf` | **No** | GT PDFs for local eval only |

## Local setup (after clone)

Copy sample files from your secure storage (not included in this repo):

```bash
cp /path/to/sample_case.zip tasks/cms/
cp /path/to/form_g28.json tasks/cms/
cp /path/to/form_i140.json tasks/cms/
cp /path/to/form_i907.json tasks/cms/
```

Or use your own case `.zip` and generate GT JSON with `apps/api` scoring scripts.

## CLI example

```bash
cd apps/api
uv run python -m modal run modal_cms_app.py \
  --case-zip ../../tasks/cms/sample_case.zip \
  --form-pdf ../../tasks/cms/blank/i-140.pdf \
  --gt-json ../../tasks/cms/form_i140.json
```

GT JSON is **scoring-only** — never sent to the agent sandbox.
