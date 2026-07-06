"""Coastline web UI — the FastAPI dashboard, served by the ``coastline-ui`` command."""

from __future__ import annotations

import os


def main() -> None:
    """Launch the dashboard (``coastline-ui``). Host/port via COASTLINE_UI_HOST/PORT."""
    import uvicorn

    host = os.environ.get("COASTLINE_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("COASTLINE_UI_PORT", "8000"))
    uvicorn.run("coastline.ui.app:app", host=host, port=port)


__all__ = ["main"]
