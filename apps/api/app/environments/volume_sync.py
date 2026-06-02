"""Modal workspace volume sync for env sandboxes (bundled under /env)."""

from __future__ import annotations

from pathlib import Path

WORKSPACE_VOLUME_NAME = "form-filling-app-workspaces"
WORKSPACE_MOUNT = Path("/vol/workspaces")


def reload_workspace_volume_if_needed(case_dir: Path) -> None:
    """
    Env sandboxes mount the workspace volume at startup; writers in other
    containers must ``commit()`` and readers must ``reload()`` before listing
    newly uploaded case files.
    """
    try:
        resolved = case_dir.resolve()
    except OSError:
        resolved = case_dir
    if not str(resolved).startswith(str(WORKSPACE_MOUNT)):
        return
    try:
        import modal

        modal.Volume.from_name(WORKSPACE_VOLUME_NAME).reload()
    except Exception:
        pass
