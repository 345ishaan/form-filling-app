"""Form Filling App API — CMS case + immigration form filling."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.anthropic import configure_anthropic_on_startup, is_anthropic_configured
from app.config.runtime import agent_runs_locally, get_sandbox_mode, is_modal_runtime
from app.routers.ws import router as ws_router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_anthropic_on_startup()
    yield


app = FastAPI(
    title="Form Filling App API",
    description="Immigration case management and USCIS form filling agent.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.get("/")
async def root():
    return {
        "status": "operational",
        "message": "Form Filling App API",
        "version": "0.1.0",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "sim_types": ["cms"],
        "services": {
            "api": "operational",
            "sandbox_mode": get_sandbox_mode(),
            "agent_runtime": "modal_inline" if agent_runs_locally() else "modal_remote",
            "api_on_modal": is_modal_runtime(),
            "anthropic": (
                "configured"
                if is_anthropic_configured()
                else ("delegated_to_modal" if not agent_runs_locally() else "missing_api_key")
            ),
        },
    }
