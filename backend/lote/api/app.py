"""FastAPI app served by the Modal web function (ARCHITECTURE.md > Web function)."""

import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

RUNS_DIR = Path("/runs")  # Modal Volume `lote-runs` mount
UI_DIR = Path("/assets/ui")  # built frontend (frontend/dist), baked into the web image


def create_app(volume=None, progress=None) -> FastAPI:
    """`volume` and `progress` are the Modal Volume and Dict; None outside Modal (tests)."""
    app = FastAPI(title="Lote")

    @app.get("/api/health")
    def health() -> dict:
        checks: dict[str, object] = {"secret": bool(os.environ.get("MODEL_API_KEY"))}
        if progress is not None:
            stamp = time.time()
            progress.put("health", stamp)
            checks["dict"] = progress.get("health") == stamp
        if volume is not None:
            volume.reload()
            checks["volume"] = RUNS_DIR.is_dir()
        checks["ui"] = (UI_DIR / "tool" / "index.html").is_file()
        return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks}

    if UI_DIR.is_dir():
        app.mount("/player", StaticFiles(directory=UI_DIR / "player", html=True), name="player")
        app.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="assets")

        @app.get("/")
        def tool_ui() -> FileResponse:
            return FileResponse(UI_DIR / "tool" / "index.html")

    return app
