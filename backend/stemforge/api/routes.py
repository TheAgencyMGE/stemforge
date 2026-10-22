"""HTTP routes. Long-running work is only ever enqueued here, never executed in-request."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from .. import __version__, queue, services
from ..config import get_settings
from ..db import get_db
from ..models import Artifact, Job, Project, Track
from ..pipeline import audio as audio_mod
from ..pipeline.instruments import EXCLUDED_MODELS, families, get_instrument
from ..pipeline.transcription import ENGINES
from ..schemas import (
    EngineOut,
    InstrumentCatalog,
    InstrumentOut,
    JobOut,
    ModelStatus,
    NotesDoc,
    NotesUpdate,
    ProjectDetail,
    ProjectPatch,
    ProjectSummary,
    RetranscribeIn,
    SystemInfo,
    TrackOut,
    TrackPatch,
)
from ..security import (
    AUDIO_FORMATS,
    display_name_from_filename,
    safe_filename,
    sniff_audio_format,
    verify_signature,
)
from ..storage import get_storage, sha256_file
from .app import ApiError

log = logging.getLogger(__name__)
router = APIRouter()
DB = Annotated[Session, Depends(get_db)]
NATIVE_FORMATS = ["wav", "flac", "mp3", "ogg", "aiff"]


# ------------------------------------------------------------------ health / system


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/readyz")
def readyz(db: DB) -> dict:
    checks: dict[str, str] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {exc}"
    try:
        get_storage().exists("projects/readiness-probe")
        checks["storage"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["storage"] = f"error: {exc}"
    from ..model_registry import status

    checks["model_htdemucs"] = "ok" if status("htdemucs")["installed"] else "missing (run stemforge-models download htdemucs)"
    ok = all(v == "ok" for v in checks.values())
    if not ok:
        raise ApiError(503, "not_ready", json.dumps(checks))
    return {"status": "ready", "checks": checks}


_DEVICE_CACHE: dict[str, str | None] = {}


def _device() -> tuple[str, str | None]:
    if "device" not in _DEVICE_CACHE:
        try:
            import torch

            pref = get_settings().device
            if pref != "cpu" and torch.cuda.is_available():
                _DEVICE_CACHE.update(device="cuda", name=torch.cuda.get_device_name(0))
            else:
                _DEVICE_CACHE.update(device="cpu", name=None)
        except Exception:  # noqa: BLE001
            _DEVICE_CACHE.update(device="unknown", name=None)
    return _DEVICE_CACHE["device"] or "unknown", _DEVICE_CACHE.get("name")


def _m4a_supported() -> bool:
    return audio_mod._try_import_av() is not None or audio_mod._ffmpeg() is not None


@router.get("/system", response_model=SystemInfo)
def system(db: DB) -> SystemInfo:
    from ..model_registry import manifest, status

    settings = get_settings()
    device, name = _device()
    counts = dict(db.execute(select(Job.status, func.count()).group_by(Job.status)).all())
    m4a = _m4a_supported()
    return SystemInfo(
        version=__version__, device=device, device_name=name,
        models=[ModelStatus(**status(n)) for n in manifest()],
        engines=[EngineOut(id=e.id, label=e.label, polyphonic=e.polyphonic, description=e.description,
                           defaults=e.defaults) for e in ENGINES.values()],
        formats=NATIVE_FORMATS + (["m4a", "webm"] if m4a else []), m4a_supported=m4a,
        max_upload_mb=settings.max_upload_mb, max_duration_seconds=settings.max_duration_seconds,
        retention_days=settings.retention_days, auth_required=bool(settings.api_token),
        queue={k: int(v) for k, v in counts.items()},
    )


# ------------------------------------------------------------------ projects


def _get_project(db: Session, project_id: str) -> Project:
    p = db.get(Project, project_id)
    if p is None:
        raise ApiError(404, "not_found", "project not found")
    return p


def _active_job(db: Session, project_id: str) -> Job | None:
    return db.scalars(select(Job).where(Job.project_id == project_id, Job.status.in_(["queued", "running"]))).first()


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(db: DB, limit: int = Query(100, ge=1, le=500)) -> list[ProjectSummary]:
    rows = db.scalars(select(Project).order_by(Project.created_at.desc()).limit(limit)).all()
    return [services.project_list_item(p) for p in rows]


@router.get("/projects/{project_id}", response_model=ProjectDetail)
def get_project(project_id: str, db: DB) -> ProjectDetail:
    return services.project_detail(db, _get_project(db, project_id))


class UploadOptions(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    six_stem: bool = False
    drum_kit: bool = False
    transcribe_mix: bool = False
    compare_engines: bool = False
    trim_start: float = Field(default=0.0, ge=0)
    trim_end: float | None = Field(default=None, gt=0)


async def _receive_upload(request: Request, dest: Path, limit: int) -> int:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise ApiError(413, "too_large", f"file exceeds the {limit // (1024 * 1024)} MB upload limit")
    size = 0
    with dest.open("wb") as out:
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise ApiError(413, "too_large", f"file exceeds the {limit // (1024 * 1024)} MB upload limit")
            out.write(chunk)
    if size == 0:
        raise ApiError(400, "empty_upload", "the uploaded file is empty")
    return size


def _check_quota(incoming: int) -> None:
    settings = get_settings()
    used = get_storage().total_bytes()
    if used + incoming * 6 > settings.storage_quota_gb * 1024**3:  # stems + previews ~ 6x the source
        raise ApiError(507, "quota_exceeded", "storage quota reached; delete old projects to free space")


def _create_project_from_file(db: Session, tmp: Path, filename: str, opts: UploadOptions, is_sample: bool = False) -> Project:
    settings = get_settings()
    with tmp.open("rb") as f:
        head = f.read(64)
    fmt = sniff_audio_format(head)
    if fmt is None:
        raise ApiError(415, "unsupported_format",
                       "not a recognised audio file. Supported: WAV, FLAC, MP3, OGG/Opus, AIFF"
                       + (", M4A/AAC" if _m4a_supported() else ""))
    if fmt in ("m4a", "webm") and not _m4a_supported():
        raise ApiError(415, "decoder_unavailable",
                       f"{fmt.upper()} needs FFmpeg, which is not available on this server. "
                       "Convert to WAV, FLAC or MP3 and try again.")
    try:
        info = audio_mod.probe(tmp, fmt)
    except audio_mod.AudioDecodeError as exc:
        raise ApiError(422, "undecodable", f"could not read the audio: {exc}") from exc
    if info.channels > settings.max_channels or info.channels < 1:
        raise ApiError(422, "bad_channels", f"{info.channels} channels is not supported")
    if not 8000 <= info.sample_rate <= 384000:
        raise ApiError(422, "bad_sample_rate", f"sample rate {info.sample_rate} Hz is not supported")
    end = opts.trim_end if opts.trim_end is not None else info.duration
    end = min(end, info.duration)
    if opts.trim_start >= end:
        raise ApiError(422, "bad_trim", "trim start must be before trim end and inside the file")
    processed = end - opts.trim_start
    if processed > settings.max_duration_seconds:
        raise ApiError(413, "too_long",
                       f"the selected audio is {processed / 60:.1f} min; the limit is "
                       f"{settings.max_duration_seconds / 60:.0f} min. Trim a section to process.")
    if processed < settings.min_duration_seconds:
        raise ApiError(422, "too_short", "audio must be at least one second long")

    clean = safe_filename(filename, default=f"audio.{fmt}")
    p = Project(
        name=(opts.name or display_name_from_filename(clean))[:200], original_filename=clean, source_format=fmt,
        source_size=tmp.stat().st_size, status="queued", is_sample=is_sample,
        options={"six_stem": opts.six_stem, "drum_kit": opts.drum_kit,
                 "transcribe_mix": opts.transcribe_mix, "compare_engines": opts.compare_engines,
                 "trim_start": opts.trim_start,
                 "trim_end": opts.trim_end if opts.trim_end and opts.trim_end < info.duration else None,
                 "source_duration": round(info.duration, 3), "source_sample_rate": info.sample_rate,
                 "source_channels": info.channels},
    )
    db.add(p)
    db.flush()
    key = f"projects/{p.id}/source/original.{fmt}"
    size = get_storage().put_file(key, tmp, AUDIO_FORMATS.get(fmt, "application/octet-stream"))
    db.add(Artifact(project_id=p.id, kind="source", storage_key=key, filename=clean,
                    content_type=AUDIO_FORMATS.get(fmt, "application/octet-stream"), size=size,
                    sha256=sha256_file(tmp)))
    try:
        queue.enqueue(db, p.id, "process", {"six_stem": opts.six_stem})
    except queue.QueueFull as exc:
        db.rollback()
        get_storage().delete_prefix(f"projects/{p.id}")
        raise ApiError(503, "queue_full", str(exc)) from exc
    db.commit()
    db.refresh(p)
    return p


def _upload_dir() -> Path:
    d = get_settings().data_dir / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("/projects", response_model=ProjectDetail, status_code=201)
async def upload_project(
    request: Request, db: DB,
    name: str | None = Query(None, max_length=200), six_stem: bool = Query(False),
    drum_kit: bool = Query(False), transcribe_mix: bool = Query(False),
    compare_engines: bool = Query(False),
    trim_start: float = Query(0.0, ge=0), trim_end: float | None = Query(None, gt=0),
) -> ProjectDetail:
    """Upload raw audio bytes as the request body (streamed to disk with a size cap).

    The original filename is sent URL-encoded in the ``X-Filename`` header.
    """
    settings = get_settings()
    filename = unquote(request.headers.get("x-filename", "audio"))
    opts = UploadOptions(name=name, six_stem=six_stem, drum_kit=drum_kit, transcribe_mix=transcribe_mix,
                         compare_engines=compare_engines, trim_start=trim_start, trim_end=trim_end)
    declared = request.headers.get("content-length")
    if declared and declared.isdigit():
        await run_in_threadpool(_check_quota, int(declared))
    tmp = _upload_dir() / f"{uuid.uuid4().hex}.part"
    try:
        await _receive_upload(request, tmp, settings.max_upload_bytes)
        p = await run_in_threadpool(_create_project_from_file, db, tmp, filename, opts)
        return services.project_detail(db, p)
    finally:
        tmp.unlink(missing_ok=True)


class SampleIn(BaseModel):
    six_stem: bool = False


@router.post("/projects/sample", response_model=ProjectDetail, status_code=201)
def create_sample(db: DB, body: SampleIn | None = None) -> ProjectDetail:
    sample = get_settings().samples_dir / "stemforge-demo.flac"
    if not sample.is_file():
        raise ApiError(404, "sample_missing", "sample audio is not installed (samples/stemforge-demo.flac)")
    tmp = _upload_dir() / f"{uuid.uuid4().hex}.part"
    shutil.copyfile(sample, tmp)
    try:
        opts = UploadOptions(name="Stemforge demo (CC0)", six_stem=bool(body and body.six_stem))
        p = _create_project_from_file(db, tmp, sample.name, opts, is_sample=True)
        return services.project_detail(db, p)
    finally:
        tmp.unlink(missing_ok=True)


@router.patch("/projects/{project_id}", response_model=ProjectDetail)
def patch_project(project_id: str, body: ProjectPatch, db: DB) -> ProjectDetail:
    p = _get_project(db, project_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(p, field, value)
    db.commit()
    db.refresh(p)
    return services.project_detail(db, p)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, db: DB) -> None:
    p = _get_project(db, project_id)
    job = _active_job(db, project_id)
    if job:
        queue.request_cancel(db, job)
    services.delete_project(db, p)


class ReprocessIn(BaseModel):
    six_stem: bool | None = None
    drum_kit: bool | None = None
    transcribe_mix: bool | None = None
    compare_engines: bool | None = None


@router.get("/instruments", response_model=InstrumentCatalog)
def list_instruments() -> InstrumentCatalog:
    return InstrumentCatalog(families=families(), excluded_models=EXCLUDED_MODELS)


@router.get("/instruments/{instrument_id}", response_model=InstrumentOut)
def instrument_detail(instrument_id: str) -> InstrumentOut:
    inst = get_instrument(instrument_id)
    if inst is None:
        raise ApiError(
            404, "unknown_instrument",
            f"No instrument {instrument_id!r}. Separation and transcription are limited to the catalog at /api/instruments. "
            "Nothing is inferred for an unknown name.",
        )
    return InstrumentOut(**inst.to_public())


@router.post("/projects/{project_id}/reprocess", response_model=JobOut, status_code=202)
def reprocess(project_id: str, body: ReprocessIn, db: DB) -> JobOut:
    """Re-run the pipeline (e.g. to add 6-stem guitar/piano). Finished stages are reused."""
    p = _get_project(db, project_id)
    if _active_job(db, project_id):
        raise ApiError(409, "busy", "this project is already processing")
    opts = dict(p.options or {})
    for key in ("six_stem", "drum_kit", "transcribe_mix", "compare_engines"):
        value = getattr(body, key)
        if value is not None:
            opts[key] = value
    p.options = opts
    p.status = "queued"
    try:
        job = queue.enqueue(db, p.id, "process", {"six_stem": (p.options or {}).get("six_stem", False)})
    except queue.QueueFull as exc:
        raise ApiError(503, "queue_full", str(exc)) from exc
    db.commit()
    return services.job_out(job)


# ------------------------------------------------------------------ jobs


def _get_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise ApiError(404, "not_found", "job not found")
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: DB) -> JobOut:
    return services.job_out(_get_job(db, job_id))


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, db: DB) -> JobOut:
    job = _get_job(db, job_id)
    queue.request_cancel(db, job)
    if job.status == "cancelled" and job.kind == "process":
        p = db.get(Project, job.project_id)
        if p:
            p.status = "cancelled"
            db.commit()
    db.refresh(job)
    return services.job_out(job)


@router.post("/jobs/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: str, db: DB) -> JobOut:
    job = _get_job(db, job_id)
    if job.status not in ("failed", "cancelled"):
        raise ApiError(409, "not_retryable", f"job is {job.status}")
    if _active_job(db, job.project_id):
        raise ApiError(409, "busy", "another job for this project is active")
    queue.requeue_for_resume(db, job)
    p = db.get(Project, job.project_id)
    if p and job.kind == "process":
        p.status = "queued"
        p.error = None
        db.commit()
    return services.job_out(job)


# ------------------------------------------------------------------ tracks / notes


def _get_track(db: Session, project_id: str, track_id: str) -> Track:
    t = db.get(Track, track_id)
    if t is None or t.project_id != project_id:
        raise ApiError(404, "not_found", "track not found")
    return t


@router.get("/projects/{project_id}/tracks/{track_id}/notes", response_model=NotesDoc)
def get_notes(project_id: str, track_id: str, db: DB) -> NotesDoc:
    t = _get_track(db, project_id, track_id)
    return NotesDoc(version=t.notes_version, notes=t.notes or [])


@router.put("/projects/{project_id}/tracks/{track_id}/notes", response_model=NotesDoc)
def put_notes(project_id: str, track_id: str, body: NotesUpdate, db: DB) -> NotesDoc:
    t = _get_track(db, project_id, track_id)
    if body.base_version != t.notes_version:
        raise ApiError(409, "version_conflict",
                       f"notes changed elsewhere (server version {t.notes_version}); reload to continue")
    ids = [n.id for n in body.notes]
    if len(ids) != len(set(ids)):
        raise ApiError(422, "duplicate_ids", "note ids must be unique")
    t.notes = [n.model_dump() for n in body.notes]
    t.notes_version += 1
    db.commit()
    return NotesDoc(version=t.notes_version, notes=t.notes)


@router.post("/projects/{project_id}/tracks/{track_id}/reset", response_model=NotesDoc)
def reset_notes(project_id: str, track_id: str, db: DB) -> NotesDoc:
    t = _get_track(db, project_id, track_id)
    t.notes = list(t.generated_notes or [])
    t.notes_version += 1
    db.commit()
    return NotesDoc(version=t.notes_version, notes=t.notes)


@router.patch("/projects/{project_id}/tracks/{track_id}", response_model=TrackOut)
def patch_track(project_id: str, track_id: str, body: TrackPatch, db: DB) -> TrackOut:
    t = _get_track(db, project_id, track_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(t, field, value)
    db.commit()
    return services.track_out(t)


@router.post("/projects/{project_id}/tracks/{track_id}/transcribe", response_model=JobOut, status_code=202)
def retranscribe(project_id: str, track_id: str, body: RetranscribeIn, db: DB) -> JobOut:
    t = _get_track(db, project_id, track_id)
    if _active_job(db, project_id):
        raise ApiError(409, "busy", "wait for the current job on this project to finish")
    engine = body.engine or t.engine
    if engine == "drum-nmf" and not t.is_drum:
        raise ApiError(422, "bad_engine", "the drum engine only applies to the drum stem")
    if t.is_drum and engine != "drum-nmf":
        raise ApiError(422, "bad_engine", "drum stems use the drum engine")
    try:
        job = queue.enqueue(db, project_id, "retranscribe",
                            {"track_id": t.id, "engine": body.engine, "params": body.params}, priority=50)
    except queue.QueueFull as exc:
        raise ApiError(503, "queue_full", str(exc)) from exc
    db.commit()
    return services.job_out(job)


# ------------------------------------------------------------------ exports / import


def _cleanup_tmpdir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


@router.get("/projects/{project_id}/export/midi")
def export_midi(project_id: str, db: DB, background: BackgroundTasks, stems: str | None = None) -> FileResponse:
    """MIDI rendered from the *current edited* notes, validated with an independent parser."""
    p = _get_project(db, project_id)
    tracks = [t for t in services.select_tracks(p, stems) if t.transcription_status == "done"]
    if not tracks:
        raise ApiError(404, "nothing_to_export", "no transcribed tracks selected")
    td = Path(tempfile.mkdtemp(prefix="sf-export-"))
    background.add_task(_cleanup_tmpdir, td)
    suffix = tracks[0].stem if len(tracks) == 1 else "multitrack"
    out = td / f"{services._slug(p.name)}-{suffix}.mid"
    try:
        services.build_export_midi(p, tracks, out)
    except Exception as exc:
        raise ApiError(500, "midi_invalid", f"MIDI validation failed: {exc}") from exc
    return FileResponse(out, media_type="audio/midi", filename=out.name)


@router.get("/projects/{project_id}/export/bundle")
def export_bundle(
    project_id: str, db: DB, background: BackgroundTasks, stems: str | None = None,
    include_stems: bool = True, include_midi: bool = True, full: bool = False,
) -> FileResponse:
    p = _get_project(db, project_id)
    if p.status != "ready":
        raise ApiError(409, "not_ready", "the project has not finished processing")
    tracks = services.select_tracks(p, stems)
    if not tracks:
        raise ApiError(404, "nothing_to_export", "no tracks selected")
    td = Path(tempfile.mkdtemp(prefix="sf-bundle-"))
    background.add_task(_cleanup_tmpdir, td)
    kind = "project" if full else "export"
    out = td / f"{services._slug(p.name)}-{kind}.zip"
    services.write_bundle(db, p, out, tracks, include_stems or full, include_midi, full)
    return FileResponse(out, media_type="application/zip", filename=out.name)


@router.post("/projects/import", response_model=ProjectDetail, status_code=201)
async def import_project(request: Request, db: DB) -> ProjectDetail:
    settings = get_settings()
    tmp = _upload_dir() / f"{uuid.uuid4().hex}.zip.part"
    try:
        await _receive_upload(request, tmp, settings.max_upload_bytes * 8)
        try:
            p = await run_in_threadpool(services.import_bundle, db, tmp)
        except services.BundleError as exc:
            db.rollback()
            raise ApiError(422, "bad_bundle", str(exc)) from exc
        return services.project_detail(db, p)
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------ signed file downloads


@router.get("/files/{artifact_id}")
def get_file(artifact_id: str, exp: int, sig: str, db: DB, download: bool = False):
    settings = get_settings()
    if not verify_signature(settings.effective_signing_secret(), artifact_id, exp, sig):
        raise ApiError(403, "bad_signature", "download link is invalid or expired; reload the project")
    art = db.get(Artifact, artifact_id)
    if art is None:
        raise ApiError(404, "not_found", "file not found")
    storage = get_storage()
    headers = {"cache-control": "private, max-age=3600"}
    if hasattr(storage, "file_path"):
        path = storage.file_path(art.storage_key)  # type: ignore[attr-defined]
        if not path.is_file():
            raise ApiError(404, "not_found", "file missing from storage")
        return FileResponse(path, media_type=art.content_type, headers=headers,
                            filename=art.filename if download else None,
                            content_disposition_type="attachment" if download else "inline")
    body = storage.open(art.storage_key)
    return StreamingResponse(iter(lambda: body.read(1 << 16), b""), media_type=art.content_type, headers=headers)
