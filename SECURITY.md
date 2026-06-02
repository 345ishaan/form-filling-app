# Sensitive data

Do **not** commit:

- **API keys** — use Modal secret `ant-key` and local `apps/api/.env` (gitignored)
- **Case documents** — `tasks/cms/sample_case.zip`, extracted `sample_case/`, or any client case uploads
- **Ground truth** — `tasks/cms/form_*.json`, `*_gt.pdf` (filled forms with names/addresses)
- **Run artifacts** — transcripts, filled exports under `~/.form-filling-app/`

Safe to commit:

- Application source code
- Blank sample form PDFs under `tasks/cms/blank/`
- `.env.example` files (empty placeholders only)

Before push:

```bash
git status
git ls-files tasks/cms/   # should only show blank/*.pdf and README.md
```
