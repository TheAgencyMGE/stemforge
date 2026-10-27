"""Pydantic API contract (exported as OpenAPI; frontend types mirror these)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Note(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, max_length=40)
    pitch: int = Field(ge=0, le=127)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    velocity: int = Field(ge=1, le=127)
    # None = authored by the user (no model confidence exists for it).
    confidence: float | None = Field(default=None, ge=0, le=1)
    bends: list[tuple[float, float]] | None = None
    # Set when a review step changed the pitch. The editor can put this pitch back.
    raw_pitch: int | None = Field(default=None, ge=0, le=127)
    flags: list[str] | None = None

    @model_validator(mode="after")
    def _order(self) -> Note:
        if self.end <= self.start:
            raise ValueError("note end must be after start")
        if self.end - self.start > 600:
            raise ValueError("note longer than 10 minutes")
        return self

    @field_validator("bends")
    @classmethod
    def _bends(cls, v):
        if v is None:
            return v
        if len(v) > 2000:
            raise ValueError("too many bend points")
        return [(float(t), max(-2.0, min(2.0, float(s)))) for t, s in v]

    @field_validator("flags")
    @classmethod
    def _flags(cls, v):
        if v is None:
            return v
        if len(v) > 8:
            raise ValueError("too many note flags")
        cleaned = []
        for item in v:
            text = str(item)
            if len(text) > 40 or not text.replace("_", "").isalnum():
                raise ValueError("invalid note flag")
            cleaned.append(text)
        return cleaned


class NotesDoc(BaseModel):
    version: int
    notes: list[Note]


class NotesUpdate(BaseModel):
    base_version: int
    notes: list[Note] = Field(max_length=100_000)


class ArtifactOut(BaseModel):
    id: str
    kind: str
    stem: str | None
    filename: str
    content_type: str
    size: int
    url: str
    meta: dict[str, Any] = {}


class TrackOut(BaseModel):
    id: str
    stem: str
    display_name: str
    order_index: int
    source_model: str
    is_drum: bool
    gm_program: int
    stats: dict[str, Any]
    warnings: list[str]
    engine: str | None
    engine_reason: str | None
    engine_params: dict[str, Any]
    notes_version: int
    note_count: int
    edited: bool
    transcription_status: str
    transcription_summary: dict[str, Any] | None
    allowed_engines: list[str] = []


class StageOut(BaseModel):
    status: str
    progress: float = 0
    message: str | None = None
    seconds: float | None = None


class JobOut(BaseModel):
    id: str
    project_id: str
    kind: str
    status: str
    stage: str | None
    progress: float
    eta_seconds: float | None
    stages: dict[str, Any]
    stage_order: list[str]
    message: str | None
    error: str | None
    attempts: int
    max_attempts: int
    params: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ProjectSummary(BaseModel):
    id: str
    name: str
    status: str
    original_filename: str
    duration: float | None
    tempo: float | None
    key: str | None
    stems: int
    is_sample: bool
    error: str | None
    created_at: datetime
    expires_at: datetime | None


class ProjectDetail(ProjectSummary):
    options: dict[str, Any]
    analysis: dict[str, Any] | None
    tempo_bpm: float | None
    time_signature: str
    offset_seconds: float
    tracks: list[TrackOut]
    artifacts: list[ArtifactOut]
    job: JobOut | None


class ProjectPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    tempo_bpm: float | None = Field(default=None, ge=20, le=400)
    time_signature: Literal["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"] | None = None


class TrackPatch(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=64)
    gm_program: int | None = Field(default=None, ge=0, le=127)


class RetranscribeIn(BaseModel):
    engine: str | None = Field(default=None, max_length=32)  # checked against the engine registry
    params: dict[str, float | None] = {}


class InstrumentOut(BaseModel):
    id: str
    family: str
    name: str
    separate: str
    transcribe: str
    stem: str | None
    engines: list[str]
    gm_program: int | None
    gm_note: int | None
    pitch_min: int | None
    pitch_max: int | None
    model: str | None
    limitation: str
    summary: str


class InstrumentCatalog(BaseModel):
    families: list[dict[str, Any]]
    excluded_models: list[dict[str, str]]


class EngineOut(BaseModel):
    id: str
    label: str
    polyphonic: bool
    description: str
    defaults: dict[str, Any]
    external: bool = False


class ModelStatus(BaseModel):
    name: str
    installed: bool
    sources: list[str]
    size_bytes: int
    license: str
    caveat: str | None


class SystemInfo(BaseModel):
    version: str
    device: str
    device_name: str | None
    models: list[ModelStatus]
    engines: list[EngineOut]
    formats: list[str]
    m4a_supported: bool
    max_upload_mb: int
    max_duration_seconds: float
    retention_days: int
    auth_required: bool
    queue: dict[str, int]
