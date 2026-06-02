"""
Standalone CMS simulation (Modal CLI, no WebSocket).

Writes agent logs under ``/vol/workspaces/runs/<env>/<run_id>/`` on Modal (or
``~/.form-filling-app/runs/`` locally):

  - summary.json       - scores, duration, config
  - transcript.json    - full turn + tool timeline
  - agent_events.jsonl - raw SDK/WS-style events
  - final_render.txt   - last env render
  - agent_filled.json  - agent filled-form export (post-run)
  - gt_filled.json     - GT reference for scoring (never shown to the agent)
  - form_scoring.json  - precision / recall / F1 vs GT (when gt_export provided)
  - filled.json        - alias of agent_filled.json (backward compatible)
  - form_filled_*.pdf  - CMS filled AcroForm PDF(s) (copied from session forms/)
  - REPORT.md          - trajectory analysis (programmatic + LLM rubric)
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.config.settings import get_agent_model
from app.services.sandbox_client import SandboxClient
from app.services.transcript_service import (
    events_path,
    save_session_transcript,
    transcript_path,
)
from app.services.workspace import commit_workspace_volume_sync, get_forms_dir, workspace_root

CMS_DEFAULT_PROMPT = (
    "You have access to an uploaded immigration case in the CMS environment. "
    "Use search_documents and read_document to inspect the case. "
    "Summarize the beneficiary, petitioner, and requested classification in concise bullets. "
    "Do not invent facts."
)

CMS_FILL_FORM_PROMPT = (
    "Fill the uploaded immigration form from the case documents. "
    "Start by calling list_uploaded_forms, then parse_form for the uploaded PDF, "
    "then use next_form_batch and submit_form_batch until the form is complete. "
    "When done, call export_filled_pdf to save form_filled_<form_type>.pdf under forms/. "
    "Do not ask for confirmation. If a value is not supported by the case documents, leave it blank."
)


def resolve_gt_json_path(
    form_pdf: Path | None,
    gt_json: str | Path | None = None,
) -> Path | None:
    """Resolve ``tasks/cms/form_<type>.json`` from blank template or explicit path."""
    if gt_json:
        p = Path(gt_json)
        return p if p.is_file() else None
    if form_pdf is None or not form_pdf.is_file():
        return None
    from app.services.form_loader import infer_form_type

    form_type, _ = infer_form_type(form_pdf)
    cms_dir = form_pdf.parent.parent if form_pdf.parent.name == "blank" else form_pdf.parent
    for candidate in (
        cms_dir / f"form_{form_type}.json",
        form_pdf.parent / f"form_{form_type}.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _gt_snapshot_for_scoring(
    gt_export: dict[str, Any],
    *,
    form_type: str | None = None,
) -> dict[str, Any]:
    """Subset of GT export for run artifacts (never written to agent workspace)."""
    forms = gt_export.get("forms")
    if not isinstance(forms, dict):
        return gt_export
    if form_type:
        ft = form_type.strip().lower()
        if ft in forms:
            return {
                "forms": {ft: forms[ft]},
                "schema_pdf": gt_export.get("schema_pdf"),
                "values_pdf": gt_export.get("values_pdf"),
            }
    return {
        "forms": dict(forms),
        "schema_pdf": gt_export.get("schema_pdf"),
        "values_pdf": gt_export.get("values_pdf"),
    }


def cms_post_run_eval(
    filled_export: dict[str, Any],
    gt_export: dict[str, Any],
    *,
    levenshtein_threshold: int = 2,
) -> dict[str, Any]:
    """Precision/recall after agent run (GT stays outside sandbox; same as ``score_filled_json``)."""
    from app.services.form_scoring import compare_filled_to_gt, parse_filled_export

    agent_by_form = parse_filled_export(filled_export)
    gt_by_form = parse_filled_export(gt_export)
    return compare_filled_to_gt(
        agent_by_form,
        gt_by_form,
        threshold=levenshtein_threshold,
        gt_export=gt_export,
    )


def _cms_render_filled_pdf(
    *,
    filled_export: dict[str, Any],
    blank_pdf_bytes: bytes,
    run_dir: Path,
    blank_pdf_name: str = "blank_form.pdf",
    form_type: str | None = None,
) -> dict[str, Any]:
    """Render agent ``filled.json`` onto the uploaded blank PDF under ``run_dir``."""
    from app.services.filled_pdf_export import export_filled_pdfs_from_json
    from app.services.form_loader import infer_form_type

    if not blank_pdf_bytes:
        return {"error": "blank_pdf_bytes required"}
    if not filled_export:
        return {"error": "filled_export empty"}

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(blank_pdf_name or "blank_form.pdf").name
    if not safe_name.lower().endswith(".pdf"):
        safe_name = "blank_form.pdf"
    blank_path = run_dir / safe_name
    blank_path.write_bytes(blank_pdf_bytes)

    ft = (form_type or infer_form_type(blank_path)[0]).strip().lower()
    filled_path = run_dir / "filled.json"
    if not filled_path.is_file():
        filled_path.write_text(json.dumps(filled_export, indent=2, default=str), encoding="utf-8")

    try:
        result = export_filled_pdfs_from_json(
            filled_path,
            template_dir=run_dir,
            out_dir=run_dir,
            template_pdf=blank_path,
            form_type=ft,
        )
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc), "blank_template": str(blank_path), "form_type": ft}

    for row in result.get("exports") or []:
        row["run_path"] = Path(row.get("out_path", "")).name

    result["blank_template"] = str(blank_path)
    result["form_type"] = ft
    return result


def _cms_filled_pdf_share_links(run_id: str, pdf_filenames: list[str]) -> dict[str, Any]:
    """Public Modal URLs for filled PDFs (requires ``modal deploy modal_cms_app.py``)."""
    links: dict[str, str] = {}
    base_url: str | None = None
    deploy_note: str | None = None

    try:
        import modal

        fn = modal.Function.from_name("form-filling-app-cms", "cms_filled_pdf")
        base_url = fn.get_web_url()
    except Exception as exc:  # noqa: BLE001
        deploy_note = (
            f"Deploy CMS app for share links: cd apps/api && modal deploy modal_cms_app.py "
            f"({type(exc).__name__}: {exc})"
        )

    if base_url:
        for name in pdf_filenames:
            safe = Path(name).name
            links[safe] = (
                f"{base_url}?run_id={quote(run_id, safe='')}&file={quote(safe, safe='')}"
            )

    out: dict[str, Any] = {"run_id": run_id, "links": links}
    if base_url:
        out["endpoint"] = base_url
    if deploy_note:
        out["deploy_note"] = deploy_note
    return out


def _runs_root() -> Path:
    root = workspace_root() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _snapshot_cms_case_for_eval(session_id: str, run_dir: Path) -> None:
    """
    Copy case + parsed text cache into the run bundle so stage-3 eval can
    classify recall losses (missing data vs retrieval gap vs fill error).
    """
    from app.services.workspace import get_case_dir, get_session_workspace

    case_src = get_case_dir(session_id)
    if not case_src.is_dir():
        return
    case_dest = run_dir / "case"
    if not case_dest.exists():
        shutil.copytree(case_src, case_dest, dirs_exist_ok=True)
    parsed_src = get_session_workspace(session_id, "cms") / "parsed"
    if parsed_src.is_dir():
        parsed_dest = run_dir / "parsed"
        if not parsed_dest.exists():
            shutil.copytree(parsed_src, parsed_dest, dirs_exist_ok=True)


def _export_run_bundle(
    session: dict[str, Any],
    run_dir: Path,
    summary: dict[str, Any],
    *,
    extra_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist transcript + events + summary into the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_session_transcript(session, sandbox_mode=session.get("sandbox_mode", "modal"))

    sid = session["session_id"]
    if transcript_path(sid).is_file():
        shutil.copy2(transcript_path(sid), run_dir / "transcript.json")
    if events_path(sid).is_file():
        shutil.copy2(events_path(sid), run_dir / "agent_events.jsonl")

    final_render = session.get("final_render") or ""
    if final_render:
        (run_dir / "final_render.txt").write_text(final_render, encoding="utf-8")

    if extra_json:
        for filename, payload in extra_json.items():
            (run_dir / filename).write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    commit_workspace_volume_sync()

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "session_id": sid,
        "summary": summary,
        "transcript_path": str(run_dir / "transcript.json"),
        "agent_events_path": str(run_dir / "agent_events.jsonl"),
    }


async def _run_trajectory_eval_stage(
    session: dict[str, Any],
    run_dir: Path,
    *,
    bonza_scenario: str | None = None,
) -> dict[str, Any] | None:
    """Stage 3: trajectory report from transcript (no reward / GT in prompts)."""
    from app.services.trajectory_eval import (
        build_trajectory_context_from_session,
        run_trajectory_analysis,
    )

    if not (run_dir / "transcript.json").is_file():
        return None

    print("Stage 3/3: Analyzing trajectory", flush=True)
    try:
        ctx = build_trajectory_context_from_session(session)
        return await run_trajectory_analysis(run_dir, ctx)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


async def _run_agent_autonomous(
    session: dict[str, Any],
    user_message: str,
) -> dict[str, Any]:
    """One user message; SDK runs in auto mode until max_turns or the model stops."""
    from app.agents.runner import collect_agent_turn

    sandbox_id = session["sandbox_id"]
    client: SandboxClient = session["client"]
    last_info: dict[str, Any] = {}
    error: str | None = None
    started = time.time()
    print("Stage 1/3: Running agent", flush=True)

    try:
        await collect_agent_turn(session, user_message)
        last_info = await client.get_info(sandbox_id) or {}
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    return {
        "last_info": last_info,
        "error": error,
        "duration_s": round(time.time() - started, 2),
    }


async def run_cms_simulation(
    *,
    case_zip_bytes: bytes | None = None,
    case_zip_path: str | Path | None = None,
    form_pdf_bytes: bytes | None = None,
    form_pdf_path: str | Path | None = None,
    persona: str = "none",
    prompts: list[str] | None = None,
    sandbox_mode: str = "modal",
    run_id: str | None = None,
    gt_export: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from app.config.anthropic import ensure_anthropic_api_key
    from app.services.case_upload import ingest_case_into_sandbox
    from app.services.form_loader import infer_form_type
    from app.services.form_upload import save_form_pdf

    ensure_anthropic_api_key()
    run_id = run_id or f"cms-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    session_id = f"standalone-{run_id}"

    if case_zip_bytes is None and case_zip_path is not None:
        case_zip_bytes = Path(case_zip_path).read_bytes()
    if form_pdf_bytes is None and form_pdf_path is not None:
        form_pdf_bytes = Path(form_pdf_path).read_bytes()
    if not case_zip_bytes:
        raise ValueError("Provide case_zip_bytes or case_zip_path")

    client = SandboxClient(mode=sandbox_mode)
    sandbox_id = await client.create("cms", session_id=session_id)

    session: dict[str, Any] = {
        "session_id": session_id,
        "env_type": "cms",
        "sandbox_id": sandbox_id,
        "client": client,
        "persona": persona,
        "agent_mode": "play",
        "eval_mode": False,
        "claude_session_id": None,
        "transcript": [],
        "sdk_client": None,
        "sdk_connected": False,
        "interrupt_requested": False,
        "sandbox_mode": sandbox_mode,
    }

    uploaded_form: dict[str, Any] | None = None
    scoring_form_type: str | None = None
    user_message: str
    outcome: dict[str, Any] = {}
    filled_export: dict[str, Any] = {}
    form_scoring: dict[str, Any] | None = None
    filled_pdf_export: dict[str, Any] | None = None
    run_dir = _runs_root() / "cms" / run_id
    try:
        case_result = await ingest_case_into_sandbox(
            session_id,
            sandbox_id,
            client,
            zip_bytes=case_zip_bytes,
        )
        session["initial_render"] = case_result.get("render") or ""

        if form_pdf_bytes:
            form_name = Path(form_pdf_path).name if form_pdf_path else "form.pdf"
            uploaded_form = save_form_pdf(
                session_id,
                pdf_bytes=form_pdf_bytes,
                filename=form_name,
                replace_existing=True,
            )
            form_rel = uploaded_form["uploaded"]["workspace_path"]
            scoring_form_type, title = infer_form_type(Path(form_name))
            user_message = (
                f"{CMS_FILL_FORM_PROMPT} The uploaded form is `{form_rel}` "
                f"({title}, form_type `{scoring_form_type}`)."
            )
        else:
            user_message = CMS_DEFAULT_PROMPT

        msg = prompts[0] if prompts else user_message
        turn = await _run_agent_autonomous(session, msg)
        info = turn["last_info"]
        print("Stage 2/3: Computing reward", flush=True)
        pdf_export: dict[str, Any] = {}
        try:
            step = await client.step(sandbox_id, {"action_type": 10, "params": "{}"})
            pdf_export = (step.get("info") or {}).get("result") or {}
        except Exception:
            pass
        filled_export = await client.get_filled(sandbox_id)
        if pdf_export:
            filled_export["pdf_export"] = pdf_export

        # GT scoring runs here only — never loaded into the sandbox (agent cannot see GT).
        if gt_export:
            try:
                form_scoring = cms_post_run_eval(filled_export, gt_export)
            except Exception as exc:  # noqa: BLE001
                form_scoring = {"error": f"{type(exc).__name__}: {exc}"}

        if form_pdf_bytes and filled_export:
            blank_name = Path(form_pdf_path).name if form_pdf_path else "blank_form.pdf"
            filled_pdf_export = _cms_render_filled_pdf(
                filled_export=filled_export,
                blank_pdf_bytes=form_pdf_bytes,
                run_dir=run_dir,
                blank_pdf_name=blank_name,
                form_type=scoring_form_type,
            )
            pdf_names = [
                Path(row["out_path"]).name
                for row in (filled_pdf_export.get("exports") or [])
                if row.get("out_path")
            ]
            if pdf_names:
                share = _cms_filled_pdf_share_links(run_id, pdf_names)
                filled_pdf_export["share"] = share

        metrics = filled_export.get("metrics") or {}
        outcome = {
            "env_type": "cms",
            "run_id": run_id,
            "session_id": session_id,
            "persona": persona,
            "model": get_agent_model(),
            "case_file_count": case_result.get("file_count"),
            "uploaded_form": uploaded_form["uploaded"] if uploaded_form else None,
            "forms_loaded": info.get("forms_loaded"),
            "field_queue_remaining": info.get("field_queue_remaining"),
            "score": (
                form_scoring.get("score_percent")
                if form_scoring and "error" not in form_scoring
                else metrics.get("score_percent")
            ),
            "metrics": metrics,
            "form_scoring": form_scoring,
            "gt_scoring_enabled": bool(gt_export),
            "filled_pdf_export": filled_pdf_export,
            "filled_pdf_share": (filled_pdf_export or {}).get("share"),
            "filled_pdf_export_error": (filled_pdf_export or {}).get("error"),
            **turn,
        }
        session["final_render"] = await client.render(sandbox_id) or ""
    finally:
        try:
            await client.terminate(sandbox_id)
        except Exception:
            pass
        sdk = session.get("sdk_client")
        if sdk is not None:
            try:
                await sdk.disconnect()
            except Exception:
                pass

    extra_json: dict[str, Any] = {}
    if filled_export:
        extra_json["agent_filled.json"] = filled_export
        extra_json["filled.json"] = filled_export
    if gt_export:
        extra_json["gt_filled.json"] = _gt_snapshot_for_scoring(
            gt_export, form_type=scoring_form_type,
        )
    if form_scoring is not None:
        extra_json["form_scoring.json"] = form_scoring
    if filled_pdf_export is not None:
        extra_json["filled_pdf_export.json"] = filled_pdf_export

    result = _export_run_bundle(session, run_dir, outcome, extra_json=extra_json or None)
    forms_dir = get_forms_dir(session_id)
    for pdf in sorted(forms_dir.glob("form_filled_*.pdf")):
        dest = run_dir / pdf.name
        if not dest.is_file():
            shutil.copy2(pdf, dest)
    _snapshot_cms_case_for_eval(session_id, run_dir)
    traj = await _run_trajectory_eval_stage(session, run_dir)
    if traj:
        outcome["trajectory_eval"] = traj
        result["summary"] = outcome
        (run_dir / "summary.json").write_text(
            json.dumps(outcome, indent=2, default=str), encoding="utf-8",
        )
    return result


def print_run_result(result: dict[str, Any]) -> None:
    summary = result.get("summary") or {}
    env_type = summary.get("env_type") or "unknown"
    print(f"run_id: {result.get('run_id')}")
    print(f"env: {env_type}")
    if env_type == "cms":
        metrics = summary.get("metrics") or {}
        fs = summary.get("form_scoring") or {}
        if fs and "error" not in fs:
            print(
                "form_scoring precision={p} recall={r} f1={f1} "
                "tp={tp} fp={fp} fn={fn}".format(
                    p=fs.get("precision"),
                    r=fs.get("recall"),
                    f1=fs.get("f1"),
                    tp=fs.get("true_positive"),
                    fp=fs.get("false_positive"),
                    fn=fs.get("false_negative"),
                )
            )
        elif fs.get("error"):
            print(f"form_scoring error: {fs['error']}")
        print(
            "score={score} fields={filled}/{total} form={form}".format(
                score=summary.get("score"),
                filled=metrics.get("fields_filled"),
                total=metrics.get("fields_total"),
                form=(summary.get("uploaded_form") or {}).get("workspace_path"),
            )
        )
        fpe = summary.get("filled_pdf_export") or {}
        for row in fpe.get("exports") or []:
            print(
                "filled_pdf: {path} ({n} fields)".format(
                    path=row.get("out_path") or row.get("run_path"),
                    n=row.get("fields_written", "?"),
                )
            )
        if summary.get("filled_pdf_export_error"):
            print(f"filled_pdf error: {summary['filled_pdf_export_error']}")
        share = (summary.get("filled_pdf_share") or {}).get("links") or {}
        for name, url in share.items():
            print(f"filled_pdf_url ({name}): {url}")
        deploy_note = (summary.get("filled_pdf_share") or {}).get("deploy_note")
        if deploy_note and not share:
            print(deploy_note)
    print(f"run_dir: {result.get('run_dir')}")
    print(f"transcript: {result.get('transcript_path')}")
    print(f"events: {result.get('agent_events_path')}")
    traj = (summary.get("trajectory_eval") or {})
    if traj.get("report_path"):
        print(f"trajectory report: {traj['report_path']}")
    elif traj.get("error"):
        print(f"trajectory eval error: {traj['error']}")
