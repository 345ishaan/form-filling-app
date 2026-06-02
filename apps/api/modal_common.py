"""Shared Modal image builders and workspace volume for form-filling-app."""

from __future__ import annotations

from pathlib import Path

import modal

from app.config.settings import MODAL_SECRET_ANTHROPIC

_HERE = Path(__file__).resolve().parent
WORKSPACE_VOLUME_NAME = "form-filling-app-workspaces"

_GT_IGNORE = [
    "**/*_gt.pdf",
    "**/*_GT.pdf",
    "**/form_*_gt.pdf",
    "**/form_*_GT.pdf",
]


def _requirements_path() -> Path:
    for candidate in (_HERE / "requirements.txt", Path("/root/requirements.txt")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing requirements.txt (checked {_HERE} and /root)")


def _tasks_dir() -> Path:
    local = _HERE.parent.parent / "tasks"
    if local.is_dir():
        return local
    return Path("/tasks")


def _environments_dir() -> Path | None:
    local = _HERE / "app" / "environments"
    if local.is_dir():
        return local
    if Path("/env").is_dir():
        return None
    return local if local.is_dir() else None


def _pip_from_requirements() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements(
        str(_requirements_path()),
    )


def _with_liteparse_system_deps(img: modal.Image) -> modal.Image:
    return img.apt_install("libreoffice", "imagemagick", "ca-certificates", "curl")


def _mount_tasks(img: modal.Image) -> modal.Image:
    tasks = _tasks_dir()
    if not tasks.is_dir():
        return img
    return img.add_local_dir(
        str(tasks),
        remote_path="/tasks",
        copy=True,
        ignore=_GT_IGNORE,
    )


def _bundle_requirements_for_worker(img: modal.Image) -> modal.Image:
    local_req = _HERE / "requirements.txt"
    if local_req.is_file():
        return img.add_local_file(
            str(local_req),
            remote_path="/root/requirements.txt",
            copy=True,
        )
    return img


def _bundle_modal_root_modules(img: modal.Image) -> modal.Image:
    for name in ("modal_common.py",):
        path = _HERE / name
        if path.is_file():
            img = img.add_local_file(
                str(path),
                remote_path=f"/root/{name}",
                copy=True,
            )
    return img


def build_env_image() -> modal.Image:
    """Gymnasium CMS env server sandbox image."""
    img = _pip_from_requirements()
    img = _with_liteparse_system_deps(img)
    img = img.run_commands(
        'python -c "import nltk; nltk.download(\'words\', quiet=True)"',
        'python -c "import liteparse; print(\'liteparse:\', getattr(liteparse, \'__version__\', \'unknown\'))"',
    )
    img = _mount_tasks(img)
    env_src = _environments_dir()
    if env_src is not None:
        img = img.add_local_dir(str(env_src), remote_path="/env", copy=True)
    return _bundle_requirements_for_worker(img)


def build_api_image() -> modal.Image:
    """Claude Agent SDK + app code + tasks."""
    img = _pip_from_requirements()
    img = _with_liteparse_system_deps(img)
    img = img.run_commands("npm install -g @anthropic-ai/claude-code@2.1.37")
    img = _mount_tasks(img)
    app_src = _HERE / "app"
    if app_src.is_dir():
        img = img.add_local_dir(str(app_src), remote_path="/root/app", copy=True)
    img = _bundle_requirements_for_worker(img)
    return _bundle_modal_root_modules(img)


workspace_volume = modal.Volume.from_name(WORKSPACE_VOLUME_NAME, create_if_missing=True)
WORKSPACE_MOUNT = {"/vol/workspaces": workspace_volume}
anthropic_secret = modal.Secret.from_name(MODAL_SECRET_ANTHROPIC)

SIM_FUNCTION_KWARGS: dict = {
    "secrets": [anthropic_secret],
    "volumes": WORKSPACE_MOUNT,
    "cpu": 2.0,
    "memory": 4096,
    "timeout": 3600,
}

MODAL_APP_NAME = "form-filling-app"
MODAL_CMS_APP_NAME = "form-filling-app-cms"
