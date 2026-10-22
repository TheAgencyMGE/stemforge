"""ORM models: projects, tracks (with editable note documents), jobs and artifacts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    original_filename: Mapped[str] = mapped_column(String(255))
    source_format: Mapped[str] = mapped_column(String(16))
    source_size: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default="queued")  # queued|processing|ready|failed|cancelled
    # Processing options chosen at upload: six_stem, trim_start, trim_end, engines...
    options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Analysis results: duration, sample_rate, tempo, beats, key, sections, loudness, clipping...
    analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # User-adjustable musical context (tempo override, time signature, grid offset)
    tempo_bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_signature: Mapped[str] = mapped_column(String(8), default="4/4")
    # Seconds between the original file's t=0 and the processed region's t=0 (trim start).
    offset_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tracks: Mapped[list[Track]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Track.order_index"
    )
    jobs: Mapped[list[Job]] = relationship(back_populates="project", cascade="all, delete-orphan")
    artifacts: Mapped[list[Artifact]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Track(Base):
    """One separated stem and its (editable) transcription."""

    __tablename__ = "tracks"
    __table_args__ = (Index("ix_tracks_project_stem", "project_id", "stem", unique=True),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    stem: Mapped[str] = mapped_column(String(32))  # vocals|drums|bass|other|guitar|piano
    display_name: Mapped[str] = mapped_column(String(64))
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    source_model: Mapped[str] = mapped_column(String(64))  # e.g. demucs:htdemucs
    is_drum: Mapped[bool] = mapped_column(Boolean, default=False)
    gm_program: Mapped[int] = mapped_column(Integer, default=0)
    # Stem audio stats: rms_db, peak_db, silent flag, leakage estimate...
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)

    engine: Mapped[str | None] = mapped_column(String(32), nullable=True)
    engine_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    engine_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Note documents. `notes` is the user's edited copy; `generated_notes` is the raw engine output.
    notes: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    generated_notes: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    notes_version: Mapped[int] = mapped_column(Integer, default=0)
    transcription_status: Mapped[str] = mapped_column(String(24), default="pending")  # pending|done|skipped|failed
    transcription_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    project: Mapped[Project] = relationship(back_populates="tracks")


class Job(Base):
    """A durable unit of background work, claimed by workers via leases."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_claim", "status", "run_after", "priority"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32))  # process|retranscribe
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|succeeded|failed|cancelled
    priority: Mapped[int] = mapped_column(Integer, default=100)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    # Progress reporting
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # overall 0..1
    eta_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    stages: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    project: Mapped[Project] = relationship(back_populates="jobs")


class Artifact(Base):
    """A file in private storage produced from (or uploaded to) a project."""

    __tablename__ = "artifacts"
    __table_args__ = (Index("ix_artifacts_lookup", "project_id", "kind", "stem"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    # source|master|stem|stem_preview|peaks|spectrogram|midi|midi_combined
    kind: Mapped[str] = mapped_column(String(32))
    stem: Mapped[str | None] = mapped_column(String(32), nullable=True)
    storage_key: Mapped[str] = mapped_column(String(512))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    # Hash of the parameters that produced this artifact; lets resumed jobs skip finished stages.
    params_hash: Mapped[str] = mapped_column(String(64), default="")
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    project: Mapped[Project] = relationship(back_populates="artifacts")
