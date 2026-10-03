"""Coastline web UI: the FastAPI dashboard served by the ``coastline-ui`` command."""

from __future__ import annotations

import os


def main() -> None:
    """Start the dashboard (``coastline-ui``) on COASTLINE_UI_HOST and COASTLINE_UI_PORT."""
    import uvicorn

    host = os.environ.get("COASTLINE_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("COASTLINE_UI_PORT", "8000"))
    uvicorn.run("coastline.ui.app:app", host=host, port=port)


__all__ = ["main"]
