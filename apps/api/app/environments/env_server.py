"""
Environment Server — FastAPI app that wraps a Gymnasium environment for
deployment inside a Modal sandbox.

The sandbox runs this server; the backend API communicates with it via HTTP.
Dynamically loads the correct Env class based on ENV_TYPE env var.

Start: ENV_TYPE=cards uvicorn env_server:app --port 8000 --host 0.0.0.0
"""

import os
import shutil
import tempfile
import traceback
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel


# ── Env factory ─────────────────────────────────────────────────────────────

_env_instance = None
_env_type = None

# Flat modules in Modal sandbox (/env); package path when run from API locally.
ENV_REGISTRY = {
    "cms": ("cms_env", "DocProcessEnv"),
}
ENV_REGISTRY_PACKAGED = {
    "cms": "app.environments.cms_env:DocProcessEnv",
}


def _load_env(env_type: str):
    """Lazy-load the requested environment class."""
    import importlib

    if env_type not in ENV_REGISTRY:
        raise ValueError(f"Unknown env_type: {env_type}. Options: {list(ENV_REGISTRY)}")

    # Modal sandbox: modules live alongside env_server.py in /env
    mod_name, cls_name = ENV_REGISTRY[env_type]
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)
    except ModuleNotFoundError:
        spec = ENV_REGISTRY_PACKAGED[env_type]
        module_path, cls_name = spec.split(":")
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)


_RENDER_MODE = {"cms": "text"}


def get_env():
    global _env_instance
    if _env_instance is None:
        env_type = os.environ.get("ENV_TYPE", "cms")
        cls = _load_env(env_type)
        render_mode = _RENDER_MODE.get(env_type, "text")
        kwargs = {"render_mode": render_mode}
        _env_instance = cls(**kwargs)
    return _env_instance


# ── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm: load env on startup
    get_env()
    yield

app = FastAPI(title="Env Server", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    import shutil

    env = get_env()
    payload: dict = {
        "status": "ok",
        "env_type": os.environ.get("ENV_TYPE", "cards"),
        "env_class": type(env).__name__,
    }
    if os.environ.get("ENV_TYPE") == "cms":
        payload["liteparse"] = {
            "node": shutil.which("node"),
        }
        try:
            import liteparse

            payload["liteparse"]["version"] = getattr(liteparse, "__version__", "unknown")
        except Exception as exc:
            payload["liteparse"]["version_error"] = str(exc)
    return payload


class ResetOptions(BaseModel):
    seed: int | None = None
    options: dict | None = None


@app.post("/reset")
async def reset(body: ResetOptions = ResetOptions()):
    try:
        env = get_env()
        obs, info = env.reset(seed=body.seed, options=body.options)
        render = env.render()
        return {"obs": _serialize(obs), "info": info, "render": render}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_format_error(e))


class StepAction(BaseModel):
    action: dict | list | str


@app.post("/step")
async def step(body: StepAction):
    try:
        env = get_env()
        action = body.action
        # Handle array action (cards: 3 positions)
        if isinstance(action, list):
            import numpy as np
            action = np.array(action)
        # Handle dict action (bonza, cms)
        obs, reward, terminated, truncated, info = env.step(action)
        render = env.render()
        return {
            "obs": _serialize(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
            "render": render,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=_format_error(e))


@app.get("/info")
async def env_info():
    """Environment snapshot without stepping (found words, score, lists)."""
    try:
        env = get_env()
        if hasattr(env, "get_env_info"):
            return {"info": env.get_env_info()}
        return {"info": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_format_error(e))


@app.get("/valid_words")
async def valid_words():
    """Bonza: valid/invalid horizontal+vertical letter runs (NLTK)."""
    if os.environ.get("ENV_TYPE") != "bonza":
        raise HTTPException(status_code=400, detail="/valid_words is only for ENV_TYPE=bonza")
    try:
        env = get_env()
        if not hasattr(env, "get_valid_words_in_current_state"):
            raise HTTPException(status_code=501, detail="Env does not support word scan")
        return {"words": env.get_valid_words_in_current_state()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=_format_error(e))


@app.get("/render")
async def render():
    try:
        env = get_env()
        return {"render": env.render()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_format_error(e))


@app.get("/form_state")
async def form_state():
    """Export parsed forms + filled values (CMS only) for state-preserving reset."""
    if os.environ.get("ENV_TYPE") != "cms":
        raise HTTPException(status_code=400, detail="/form_state is only for ENV_TYPE=cms")
    try:
        env = get_env()
        if hasattr(env, "export_form_state"):
            return {"state": env.export_form_state()}
        return {"state": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_format_error(e))


@app.get("/filled")
async def filled():
    """Export filled form values + metrics (CMS only). Used by GET /ws/sessions/{id}/filled."""
    if os.environ.get("ENV_TYPE") != "cms":
        raise HTTPException(status_code=400, detail="/filled is only for ENV_TYPE=cms")
    try:
        env = get_env()
        if not hasattr(env, "export_filled"):
            return {"forms": {}, "metrics": {}}
        obs, _, _, _, info = env.step({"action_type": 6, "params": "{}"})
        metrics = (info or {}).get("result", {})
        return {"forms": env.export_filled(), "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=_format_error(e))


def _workspace_case_dir(session_id: str) -> Path:
    return Path("/vol/workspaces") / session_id.strip() / "case"


def _commit_workspace_volume() -> None:
    try:
        import modal

        modal.Volume.from_name("form-filling-app-workspaces").commit()
    except Exception:
        pass


@app.post("/ingest_case")
async def ingest_case(
    file: UploadFile = File(...),
    session_id: str = Query("", description="Extract into /vol/workspaces/{session_id}/case"),
):
    """Extract case zip in this sandbox and reset CMS env (persists on workspace volume when session_id set)."""
    if os.environ.get("ENV_TYPE") != "cms":
        raise HTTPException(status_code=400, detail="ingest_case is only for ENV_TYPE=cms")
    tmp: Path | None = None
    try:
        raw = await file.read()
        if not raw:
            raise ValueError("Empty upload")

        sid = session_id.strip()
        if sid:
            case_dir = _workspace_case_dir(sid)
            case_dir.parent.mkdir(parents=True, exist_ok=True)
            if case_dir.exists():
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)
            zip_path = case_dir.parent / "case.zip"
        else:
            tmp = Path(tempfile.mkdtemp(prefix="cms_case_"))
            zip_path = tmp / "upload.zip"
            case_dir = tmp / "case"
            case_dir.mkdir()

        zip_path.write_bytes(raw)
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("Not a valid .zip file")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(case_dir)

        env = get_env()
        reset_opts: dict = {"case_dir": str(case_dir), "hide_gt": True}
        if hasattr(env, "export_form_state"):
            prior = env.export_form_state()
            if prior.get("parsed_forms"):
                reset_opts["parsed_forms"] = prior["parsed_forms"]
                if prior.get("filled"):
                    reset_opts["filled"] = prior["filled"]
        obs, info = env.reset(options=reset_opts)
        render = env.render()
        file_count = sum(1 for f in case_dir.rglob("*") if f.is_file())
        if sid:
            _commit_workspace_volume()
        return {
            "status": "ok",
            "file_count": file_count,
            "case_dir": str(case_dir),
            "obs": _serialize(obs),
            "info": info,
            "render": render,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=_format_error(e))
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


@app.post("/close")
async def close():
    global _env_instance
    if _env_instance is not None:
        try:
            _env_instance.close()
        except Exception:
            pass
        _env_instance = None
    return {"status": "closed"}


def _serialize(obj):
    """Convert numpy arrays / objects to JSON-serializable types."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _format_error(e: Exception) -> str:
    return f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
