"""
Sandbox Client — manages Modal sandbox lifecycle for environment sessions.

Supports two modes:
- "modal": Creates real Modal sandboxes running env_server.
- "local": Runs environment in-process (no Modal needed for local dev).

Usage:
    client = SandboxClient(mode="local")  # or "modal"
    url = await client.create("cards")
    obs = await client.reset(url)
    obs, reward, done, info = await client.step(url, action)
    await client.terminate(url)
"""

import asyncio
import shlex
from pathlib import Path
from typing import Any, Optional

import httpx

# CMS reset can index PDFs on cold Modal sandboxes; keep read timeout generous.
_MODAL_HTTP_TIMEOUT = httpx.Timeout(15.0, read=120.0)
_LOCAL_HTTP_TIMEOUT = 30.0

# Structural env-var knobs for env_server (read by env_server._cards_kwargs).
# Add new ones here as you expose more reset-time options.
_ENV_VAR_OPTIONS = {
    "cards": {"n_suits": "N_SUITS"},
}


def _env_var_assignments(env_type: str, options: dict | None) -> str:
    """Build ``KEY=value KEY2=value2 `` prefix for env_server bash exec."""
    if not options:
        return ""
    mapping = _ENV_VAR_OPTIONS.get(env_type, {})
    parts: list[str] = []
    for opt_key, env_key in mapping.items():
        if opt_key in options and options[opt_key] is not None:
            parts.append(f"{env_key}={shlex.quote(str(options[opt_key]))}")
    return (" ".join(parts) + " ") if parts else ""


def _tunnel_url_from_mapping(tunnels: dict, port: int) -> str:
    if port not in tunnels:
        raise RuntimeError(
            f"No tunnel for port {port} (available ports: {list(tunnels.keys())})"
        )
    return tunnels[port].url


def get_sandbox_tunnel_url(sandbox: Any, port: int, timeout: int = 60) -> str:
    """Resolve public URL for a sandbox port (Modal SDK >= 0.64 uses ``tunnels()``)."""
    return _tunnel_url_from_mapping(sandbox.tunnels(timeout=timeout), port)


async def get_sandbox_tunnel_url_async(sandbox: Any, port: int, timeout: int = 60) -> str:
    """Async variant for use inside FastAPI/async request handlers."""
    return _tunnel_url_from_mapping(await sandbox.tunnels.aio(timeout=timeout), port)


class SandboxClient:
    """Manages sandbox lifecycle and proxies env API calls."""

    def __init__(self, mode: str = "local"):
        self.mode = mode
        self._local_envs: dict[str, Any] = {}  # sandbox_id -> env instance
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            timeout = _MODAL_HTTP_TIMEOUT if self.mode == "modal" else _LOCAL_HTTP_TIMEOUT
            self._http_client = httpx.AsyncClient(timeout=timeout)
        return self._http_client

    async def create(
        self,
        env_type: str,
        options: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Create a sandbox running the given environment type.
        Returns a sandbox URL or ID.
        """
        if self.mode == "modal":
            return await self._create_modal(env_type, options, session_id=session_id)
        else:
            return self._create_local(env_type, options, session_id=session_id)

    async def _create_modal(
        self,
        env_type: str,
        options: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """Spawn a Modal sandbox using the shared env image (requirements.txt)."""
        import modal
        from app.environments.env_server import ENV_REGISTRY
        from modal_common import WORKSPACE_MOUNT, build_env_image

        if env_type not in ENV_REGISTRY:
            raise ValueError(f"Unknown env_type: {env_type}")

        image = build_env_image()
        modal_app = await modal.App.lookup.aio("form-filling-app", create_if_missing=True)

        create_kwargs: dict[str, Any] = {
            "app": modal_app,
            "image": image,
            "encrypted_ports": [8000],
            "timeout": 3600,
            "cpu": 1.0,
            "memory": 2048,
            "workdir": "/env",
        }
        if env_type == "cms":
            create_kwargs["volumes"] = WORKSPACE_MOUNT
        sandbox = await modal.Sandbox.create.aio(**create_kwargs)

        env_assigns = _env_var_assignments(env_type, options)
        await sandbox.exec.aio(
            "bash", "-c",
            f"cd /env && ENV_TYPE={env_type} TASKS_ROOT=/tasks {env_assigns}"
            "uvicorn env_server:app --host 0.0.0.0 --port 8000 &",
        )

        url = await get_sandbox_tunnel_url_async(sandbox, 8000, timeout=90)
        client = await self._get_client()
        healthy = False
        for _ in range(60):
            try:
                resp = await client.get(f"{url}/health")
                if resp.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        if not healthy:
            raise RuntimeError(
                f"Modal sandbox for {env_type!r} did not respond on /health within 60s ({url})"
            )

        return url

    async def ingest_case_zip(
        self,
        sandbox_url: str,
        zip_path: str | Path,
        *,
        session_id: str = "",
    ) -> dict:
        """Stream case zip to env sandbox; extract + reset in the sandbox container."""
        client = await self._get_client()
        path = Path(zip_path)
        params = {"session_id": session_id} if session_id else None
        with path.open("rb") as f:
            resp = await client.post(
                f"{sandbox_url}/ingest_case",
                files={"file": (path.name, f, "application/zip")},
                params=params,
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"ingest_case failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()

    def _create_local(
        self,
        env_type: str,
        options: dict | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create an in-process CMS environment instance."""
        from app.environments.cms_env import DocProcessEnv

        if env_type != "cms":
            raise ValueError(f"Unknown env_type: {env_type!r} (only cms supported)")

        opts = options or {}
        case_dir = None
        if session_id:
            from app.services.workspace import get_case_dir

            cd = get_case_dir(session_id)
            if cd.is_dir() and any(cd.rglob("*")):
                case_dir = cd
        cms_init_kwargs = {"render_mode": "text", "case_dir": case_dir}
        if opts.get("forms_dir"):
            cms_init_kwargs["forms_dir"] = opts["forms_dir"]
        env = DocProcessEnv(**cms_init_kwargs)

        reset_options = opts.get("reset_options")
        cms_reset_opts: dict = dict(reset_options or {})
        for key in ("case_dir", "parsed_forms", "hide_gt"):
            if key in opts and opts[key] is not None:
                cms_reset_opts.setdefault(key, opts[key])
        reset_options = cms_reset_opts or None
        env.reset(seed=opts.get("seed"), options=reset_options)
        sandbox_id = f"local-{env_type}-{id(env)}"
        self._local_envs[sandbox_id] = env
        return sandbox_id

    async def reset(
        self,
        sandbox_id: str,
        options: dict | None = None,
        seed: int | None = None,
    ) -> dict:
        """Reset environment and return initial state."""
        if sandbox_id in self._local_envs:
            return self._local_reset(sandbox_id, options, seed=seed)
        body: dict[str, Any] = {}
        if seed is not None:
            body["seed"] = seed
        if options is not None:
            body["options"] = options
        return await self._remote_call(sandbox_id, "POST", "/reset", body)

    async def step(self, sandbox_id: str, action) -> dict:
        """Step the environment with the given action."""
        if sandbox_id in self._local_envs:
            return self._local_step(sandbox_id, action)
        else:
            return await self._remote_call(sandbox_id, "POST", "/step", {"action": action})

    async def render(self, sandbox_id: str) -> str:
        """Get the current rendered state."""
        if sandbox_id in self._local_envs:
            env = self._local_envs[sandbox_id]
            return env.render() or ""
        else:
            result = await self._remote_call(sandbox_id, "GET", "/render")
            return result.get("render", "")

    async def get_info(self, sandbox_id: str) -> dict:
        """Environment info snapshot without an invalid step."""
        if sandbox_id in self._local_envs:
            env = self._local_envs[sandbox_id]
            if hasattr(env, "get_env_info"):
                return env.get_env_info()
            return {}
        result = await self._remote_call(sandbox_id, "GET", "/info")
        return result.get("info", {})

    async def get_valid_words(self, sandbox_id: str) -> dict:
        """Bonza: NLTK-classified horizontal/vertical letter runs on current board."""
        if sandbox_id in self._local_envs:
            env = self._local_envs[sandbox_id]
            if hasattr(env, "get_valid_words_in_current_state"):
                return env.get_valid_words_in_current_state()
            return {"valid": [], "invalid": []}
        result = await self._remote_call(sandbox_id, "GET", "/valid_words")
        return result.get("words", {"valid": [], "invalid": []})

    async def get_form_state(self, sandbox_id: str) -> dict:
        """CMS parsed forms + filled values (for preserving state across case reset)."""
        if sandbox_id in self._local_envs:
            env = self._local_envs[sandbox_id]
            if hasattr(env, "export_form_state"):
                return env.export_form_state()
            return {}
        result = await self._remote_call(sandbox_id, "GET", "/form_state")
        return result.get("state", {})

    async def get_filled(self, sandbox_id: str) -> dict:
        """CMS filled-form export + metrics."""
        if sandbox_id in self._local_envs:
            env = self._local_envs[sandbox_id]
            forms = env.export_filled() if hasattr(env, "export_filled") else {}
            if hasattr(env, "_compute_metrics"):
                metrics = env._compute_metrics()
            else:
                _, _, _, _, info = env.step({"action_type": 6, "params": "{}"})
                metrics = (info or {}).get("result", {})
            return {"forms": forms, "metrics": metrics}
        return await self._remote_call(sandbox_id, "GET", "/filled")

    async def health(self, sandbox_id: str) -> bool:
        """Check if sandbox is healthy."""
        if sandbox_id in self._local_envs:
            return True
        try:
            result = await self._remote_call(sandbox_id, "GET", "/health")
            return result.get("status") == "ok"
        except Exception:
            return False

    async def terminate(self, sandbox_id: str):
        """Terminate and clean up sandbox."""
        if sandbox_id in self._local_envs:
            env = self._local_envs.pop(sandbox_id)
            try:
                env.close()
            except Exception:
                pass
        else:
            try:
                await self._remote_call(sandbox_id, "POST", "/close")
            except Exception:
                pass

    async def close(self):
        """Clean up all sandboxes."""
        for sid in list(self._local_envs.keys()):
            await self.terminate(sid)
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _local_reset(
        self,
        sandbox_id: str,
        options: dict | None = None,
        seed: int | None = None,
    ) -> dict:
        env = self._local_envs[sandbox_id]
        obs, info = env.reset(seed=seed, options=options)
        return {"obs": self._serialize(obs), "info": info, "render": env.render()}

    def _local_step(self, sandbox_id: str, action) -> dict:
        env = self._local_envs[sandbox_id]
        import numpy as np
        if isinstance(action, list):
            action = np.array(action)
        obs, reward, terminated, truncated, info = env.step(action)
        return {
            "obs": self._serialize(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
            "render": env.render(),
        }

    async def _remote_call(self, url: str, method: str, path: str, body: dict | None = None) -> dict:
        client = await self._get_client()
        full_url = f"{url}{path}"
        try:
            if method == "GET":
                resp = await client.get(full_url)
            else:
                resp = await client.post(full_url, json=body or {})
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Sandbox request timed out ({method} {path}): {type(e).__name__}"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Sandbox HTTP error ({method} {path}): {type(e).__name__}: {e!r}"
            ) from e
        if resp.status_code >= 400:
            raise RuntimeError(f"Sandbox error ({resp.status_code}): {resp.text[:500]}")
        return resp.json()

    @staticmethod
    def _serialize(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: SandboxClient._serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [SandboxClient._serialize(v) for v in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)
