"""Compatibility shim.

This repo's FastAPI instance lives in top-level `main.py` as `app`.
Some tooling/docs expect `uvicorn app.main:app`, so we re-export it here.
"""

from main import app  # noqa: F401
