"""Test fixtures: every test gets an isolated data dir (SQLite + local storage)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

BACKEND = Path(__file__).resolve().parents[1]
SAMPLES = BACKEND.parent / "samples"


def _clear_caches() -> None:
    from stemforge import config, db, storage

    config.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.get_sessionmaker.cache_clear()
    storage.get_storage.cache_clear()


@pytest.fixture()
def env(monkeypatch):
    # Short temp root: Windows MAX_PATH is easy to hit with nested artifact keys.
    root = Path(tempfile.mkdtemp(prefix="sf-"))
    monkeypatch.setenv("STEMFORGE_ENV", "test")
    monkeypatch.setenv("STEMFORGE_DATA_DIR", str(root))
    monkeypatch.setenv("STEMFORGE_MODELS_DIR", os.environ.get("STEMFORGE_MODELS_DIR", str(BACKEND / "models")))
    monkeypatch.setenv("STEMFORGE_RATE_LIMIT_PER_MINUTE", "100000")
    monkeypatch.setenv("STEMFORGE_JOB_RETRY_BASE_SECONDS", "0")
    _clear_caches()
    from stemforge.migrate import upgrade

    upgrade()
    yield root
    from stemforge.db import get_engine

    get_engine().dispose()
    _clear_caches()


@pytest.fixture()
def client(env):
    from fastapi.testclient import TestClient

    from stemforge.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def models_installed() -> bool:
    from stemforge import model_registry

    try:
        model_registry.verify("htdemucs")
        return True
    except Exception:
        return False


requires_models = pytest.mark.skipif(
    not (BACKEND / "models" / "demucs" / "htdemucs").is_dir(), reason="htdemucs weights not downloaded"
)


def write_tone(path: Path, seconds: float = 2.0, freq: float = 220.0, sr: int = 44100, channels: int = 2) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * freq * t)
    data = np.stack([x] * channels, axis=1)
    sf.write(str(path), data, sr)
    return path
