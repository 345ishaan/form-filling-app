"""Per-session filesystem workspaces for Claude SDK (Bash, Read, Write)."""

import asyncio
import shutil
import zipfile
from pathlib import Path

from app.config.runtime import is_modal_runtime

WORKSPACE_ROOT = Path.home() / ".form-filling-app" / "workspaces"
MODAL_WORKSPACE_ROOT = Path("/vol/workspaces")


def workspace_root() -> Path:
    if is_modal_runtime():
        MODAL_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        return MODAL_WORKSPACE_ROOT
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT


def get_session_workspace(session_id: str, env_type: str) -> Path:
    """Return (and create) the isolated workspace directory for a session."""
    root = workspace_root() / session_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "case").mkdir(exist_ok=True)
    (root / "parsed").mkdir(exist_ok=True)
    (root / "scripts").mkdir(exist_ok=True)
    (root / "forms").mkdir(exist_ok=True)
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text(
            f"Session workspace for form-filling-app\n"
            f"env_type: {env_type}\n"
            f"session_id: {session_id}\n\n"
            f"- case/    — extracted case documents (CMS)\n"
            f"- parsed/  — LiteParse text cache (.pdf/.docx/.jpg → .txt)\n"
            f"- forms/   — uploaded USCIS form PDFs + parsed schemas (CMS)\n"
            f"- scripts/ — helper scripts\n",
            encoding="utf-8",
        )
    return root


def get_case_dir(session_id: str) -> Path:
    """Directory where uploaded case evidence is extracted."""
    case = get_session_workspace(session_id, "cms") / "case"
    case.mkdir(parents=True, exist_ok=True)
    return case


def get_forms_dir(session_id: str) -> Path:
    """Directory where uploaded form PDFs and parsed schemas live."""
    forms = get_session_workspace(session_id, "cms") / "forms"
    forms.mkdir(parents=True, exist_ok=True)
    return forms


async def save_case_zip_bytes(session_id: str, data: bytes) -> dict:
    """Write zip to workspace, extract into ``case/``, return summary metadata."""
    if not data:
        raise ValueError("Empty upload")
    root = get_session_workspace(session_id, "cms")
    zip_path = root / "case.zip"
    zip_path.write_bytes(data)

    case_dir = get_case_dir(session_id)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("File is not a valid .zip archive")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(case_dir)

    file_count = sum(1 for f in case_dir.rglob("*") if f.is_file())
    await commit_workspace_volume()
    return {
        "session_id": session_id,
        "case_dir": str(case_dir),
        "zip_bytes": len(data),
        "file_count": file_count,
    }


WORKSPACE_VOLUME_NAME = "form-filling-app-workspaces"


async def commit_workspace_volume() -> None:
    """Persist workspace writes (call from the writer container after case/form upload)."""
    if not is_modal_runtime():
        return
    try:
        import modal

        vol = modal.Volume.from_name(WORKSPACE_VOLUME_NAME)
        await vol.commit.aio()
    except Exception:
        pass


def commit_workspace_volume_sync() -> None:
    """Sync commit for non-async callers."""
    if not is_modal_runtime():
        return
    try:
        import modal

        modal.Volume.from_name(WORKSPACE_VOLUME_NAME).commit()
    except Exception:
        pass


def reload_workspace_volume() -> None:
    """Fetch latest committed workspace data in this container (env sandboxes after upload)."""
    if not is_modal_runtime():
        return
    try:
        import modal

        modal.Volume.from_name(WORKSPACE_VOLUME_NAME).reload()
    except Exception:
        pass
