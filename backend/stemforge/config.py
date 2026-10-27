"""Environment-based configuration. Every setting can be overridden with STEMFORGE_<NAME>."""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STEMFORGE_", env_file=".env", extra="ignore")

    env: Literal["development", "production", "test"] = "development"
    data_dir: Path = BACKEND_ROOT / "data"
    database_url: str = ""  # default: sqlite file under data_dir
    samples_dir: Path = BACKEND_ROOT.parent / "samples"

    # Storage
    storage_backend: Literal["local", "s3"] = "local"
    s3_bucket: str = ""
    s3_prefix: str = "stemforge/"
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    signing_secret: str = Field(default="")  # HMAC key for local signed downloads
    signed_url_ttl_seconds: int = 3600

    # Upload / processing limits
    max_upload_mb: int = 200
    max_duration_seconds: float = 15 * 60
    min_duration_seconds: float = 1.0
    max_channels: int = 8
    storage_quota_gb: float = 50.0
    retention_days: int = 14  # 0 disables automatic deletion

    # Security
    api_token: str = ""  # if set, every /api request needs "Authorization: Bearer <token>"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    rate_limit_per_minute: int = 30  # mutating requests per client IP

    # Models / inference
    models_dir: Path = BACKEND_ROOT / "models"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    separator: Literal["demucs", "audio_separator"] = "demucs"
    demucs_model: Literal["htdemucs", "htdemucs_ft"] = "htdemucs"
    demucs_segment: float | None = None  # seconds; lower to reduce VRAM (<= 7.8 for HT models)
    demucs_shifts: int = 0  # >0 adds random time shifts (slower, non-deterministic)
    separation_chunk_seconds: float = 30.0
    # Optional research engines, each in its own virtualenv (see pipeline/external.py).
    # Let installed research engines into the automatic engine policy (off: Basic Pitch/pYIN/NMF only).
    research_engines: bool = False
    transkun_python: str = ""
    muscriptor_python: str = ""
    yourmt3_python: str = ""
    yourmt3_dir: str = ""  # checkout of the mimbres/YourMT3 space with its checkpoint
    muscriptor_model: Literal["small", "medium", "large"] = "small"  # small is the CPU-practical size
    # MIDI rendering for fidelity checks (pipeline/render.py): FluidSynth binary and an SF2/SF3 soundfont.
    fluidsynth_path: str = ""
    soundfont_path: str = ""

    # Worker / queue
    worker_concurrency: int = 1
    job_lease_seconds: int = 120
    job_heartbeat_seconds: int = 10
    job_max_attempts: int = 3
    job_retry_base_seconds: float = 15.0
    max_queued_jobs: int = 50

    @field_validator("data_dir", "models_dir", mode="after")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'stemforge.db').as_posix()}"

    @property
    def storage_dir(self) -> Path:
        return self.data_dir / "storage"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def effective_signing_secret(self) -> str:
        if self.signing_secret:
            return self.signing_secret
        # Persist a generated secret so signed URLs survive restarts in local mode.
        path = self.data_dir / ".signing_secret"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(secrets.token_hex(32))
        return path.read_text().strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
