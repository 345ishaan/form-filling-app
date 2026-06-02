"""
Standalone CMS environment — Modal app ``form-filling-app-cms``.

  cd apps/api
  uv run python -m modal run modal_cms_app.py \
      --case-zip ../../tasks/cms/sample_case.zip \
      --form-pdf ../../tasks/cms/blank/g-28.pdf \
      --gt-json ../../tasks/cms/form_g28.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import modal

from modal_common import MODAL_CMS_APP_NAME, SIM_FUNCTION_KWARGS, build_api_image, workspace_volume

app = modal.App(MODAL_CMS_APP_NAME)
api_image = build_api_image()

_DEFAULT_CASE = Path(__file__).resolve().parent.parent.parent / "tasks" / "cms" / "sample_case.zip"


@app.function(image=api_image, **SIM_FUNCTION_KWARGS)
def run_cms(payload: dict) -> dict:
    sys.path.insert(0, "/root")
    from app.config.anthropic import configure_anthropic_on_startup
    from app.services.cms_sim import run_cms_simulation

    configure_anthropic_on_startup()
    case_bytes = payload.get("case_zip_bytes")
    if case_bytes is None and payload.get("case_zip_path"):
        case_bytes = Path(payload["case_zip_path"]).read_bytes()
    if not case_bytes:
        raise ValueError("case_zip_bytes or case_zip_path required")

    form_bytes = payload.get("form_pdf_bytes")
    if form_bytes is None and payload.get("form_pdf_path"):
        form_bytes = Path(payload["form_pdf_path"]).read_bytes()

    return asyncio.run(
        run_cms_simulation(
            case_zip_bytes=case_bytes,
            form_pdf_bytes=form_bytes,
            form_pdf_path=payload.get("form_pdf_path"),
            persona=payload.get("persona", "none"),
            prompts=payload.get("prompts"),
            sandbox_mode="modal",
            run_id=payload.get("run_id"),
            gt_export=payload.get("gt_export"),
        )
    )


def _load_gt_export(form_path: Path | None, gt_json: str) -> dict | None:
    from app.services.cms_sim import resolve_gt_json_path

    gt_path = resolve_gt_json_path(form_path, gt_json or None)
    if gt_path is None:
        return None
    return json.loads(gt_path.read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(
    case_zip: str = "",
    form_pdf: str = "",
    gt_json: str = "",
    persona: str = "none",
    prompt: str = "",
):
    from app.services.cms_sim import print_run_result

    zip_path = Path(case_zip) if case_zip else _DEFAULT_CASE
    if not zip_path.is_file():
        image_path = Path("/tasks/cms/sample_case.zip")
        if image_path.is_file():
            zip_path = image_path
        else:
            raise FileNotFoundError(
                f"Case zip not found: {case_zip or _DEFAULT_CASE}. Pass --case-zip path/to/case.zip"
            )

    form_path = Path(form_pdf) if form_pdf else None
    if form_path and not form_path.is_file():
        image_form = Path("/tasks/cms/blank") / form_path.name
        if image_form.is_file():
            form_path = image_form
        else:
            raise FileNotFoundError(f"Form PDF not found: {form_pdf}")

    prompts = [prompt] if prompt else None
    gt_export = _load_gt_export(form_path, gt_json) if form_path else None
    if form_path and gt_export is None:
        print(
            "Warning: no GT JSON found — post-run precision/recall skipped. "
            "Pass --gt-json or add tasks/cms/form_<type>.json",
            file=sys.stderr,
        )

    result = run_cms.remote({
        "case_zip_bytes": zip_path.read_bytes(),
        "form_pdf_bytes": form_path.read_bytes() if form_path else None,
        "form_pdf_path": str(form_path) if form_path else None,
        "persona": persona,
        "prompts": prompts,
        "gt_export": gt_export,
    })
    print_run_result(result)


@app.function(
    image=api_image,
    volumes={"/vol/workspaces": workspace_volume},
    timeout=120,
)
@modal.fastapi_endpoint(method="GET")
def cms_filled_pdf(run_id: str, file: str = "") -> object:
    """Serve a filled PDF from ``runs/cms/<run_id>/`` on the workspace volume."""
    sys.path.insert(0, "/root")
    from fastapi.responses import Response

    from app.services.workspace import workspace_root

    workspace_volume.reload()

    if not run_id or not file:
        return Response(content="run_id and file query params required", status_code=400)

    safe = Path(file).name
    if not safe.startswith("form_filled_") or not safe.lower().endswith(".pdf"):
        return Response(content="file must be form_filled_<type>.pdf", status_code=400)

    if ".." in run_id or "/" in run_id or "\\" in run_id:
        return Response(content="invalid run_id", status_code=400)

    pdf_path = workspace_root() / "runs" / "cms" / run_id / safe
    if not pdf_path.is_file():
        return Response(content=f"not found: {run_id}/{safe}", status_code=404)

    return Response(
        content=pdf_path.read_bytes(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe}"'},
    )
