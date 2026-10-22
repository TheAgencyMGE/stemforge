"""ASGI entry point: ``uvicorn stemforge.api.main:app``."""

from .app import create_app

app = create_app()
