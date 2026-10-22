"""Application services shared by API routes: serialisation, exports, bundles, retention."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import time
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import __version__
from .config import get_settings
from .models import Artifact, Job, Project, Track, utcnow
from .pipeline import midi
from .pipeline.runner import RETRANSCRIBE_STAGES, STAGES, _slug
from .schemas import ArtifactOut, JobOut, ProjectDetail, ProjectSummary, TrackOut
from .security import sign
from .storage import get_storage, sha256_file

log = logging.getLogger(__name__)
PUBLIC_KINDS = {"master", "preview", "stem", "peaks", "spectrogram", "midi", "midi_combined"}


def signed_url(art: Artifact, download: bool = False) -> str:
    settings = get_settings()
    ttl = settings.signed_url_ttl_seconds
    if settings.storage_backend == "s3":
        url = get_storage().presigned_url(art.storage_key, art.filename, ttl)
        if url:
            return url
    exp = int(time.time()) + ttl
    sig = sign(settings.effective_signing_secret(), art.id, exp)
    return f"/api/files/{art.id}?exp={exp}&sig={sig}" + ("&download=1" if download else "")


def job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id, project_id=job.project_id, kind=job.kind, status=job.status, stage=job.stage,
        progress=job.progress, eta_seconds=job.eta_seconds, stages=job.stages or {},
        stage_order=RETRANSCRIBE_STAGES if job.kind == "retranscribe" else STAGES,
        message=job.message, error=job.error, attempts=job.attempts, max_attempts=job.max_attempts,
        params=job.params or {}, created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
    )


def track_out(t: Track) -> TrackOut:
    summary = None
    if t.transcription_detail and t.transcription_detail.startswith("{"):
        try:
            summary = json.loads(t.transcription_detail)
        except ValueError:
            summary = None
    elif t.transcription_detail:
        summary = {"detail": t.transcription_detail}
    return TrackOut(
        id=t.id, stem=t.stem, display_name=t.display_name, order_index=t.order_index,
        source_model=t.source_model, is_drum=t.is_drum, gm_program=t.gm_program, stats=t.stats or {},
        warnings=t.warnings or [], engine=t.engine, engine_reason=t.engine_reason,
        engine_params=t.engine_params or {}, notes_version=t.notes_version, note_count=len(t.notes or []),
        edited=(t.notes or []) != (t.generated_notes or []), transcription_status=t.transcription_status,
        transcription_summary=summary,
    )


def latest_job(db: Session, project_id: str) -> Job | None:
    return db.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc())).first()


def project_summary(p: Project) -> dict[str, Any]:
    a = p.analysis or {}
    return {
        "id": p.id, "name": p.name, "status": p.status, "original_filename": p.original_filename,
        "duration": a.get("duration"), "tempo": p.tempo_bpm or a.get("tempo"),
        "key": (a.get("key") or {}).get("label"), "stems": len(p.tracks), "is_sample": p.is_sample,
        "error": p.error, "created_at": p.created_at, "expires_at": p.expires_at,
    }


def project_list_item(p: Project) -> ProjectSummary:
    return ProjectSummary(**project_summary(p))


def project_detail(db: Session, p: Project) -> ProjectDetail:
    job = latest_job(db, p.id)
    arts = [
        ArtifactOut(id=a.id, kind=a.kind, stem=a.stem, filename=a.filename, content_type=a.content_type,
                    size=a.size, url=signed_url(a, download=a.kind in {"stem", "midi", "midi_combined"}),
                    meta=a.meta or {})
        for a in p.artifacts if a.kind in PUBLIC_KINDS
    ]
    analysis = dict(p.analysis or {})
    analysis.pop("_hash", None)
    return ProjectDetail(
        **project_summary(p), options=p.options or {}, analysis=analysis or None, tempo_bpm=p.tempo_bpm,
        time_signature=p.time_signature, offset_seconds=p.offset_seconds,
        tracks=[track_out(t) for t in p.tracks], artifacts=arts, job=job_out(job) if job else None,
    )


# ------------------------------------------------------------------ MIDI export from current edits


def build_export_midi(p: Project, tracks: list[Track], out: Path) -> dict:
    bpm = p.tempo_bpm or (p.analysis or {}).get("tempo") or 120.0
    specs = [midi.MidiTrackSpec(t.display_name, t.notes or [], t.gm_program, t.is_drum) for t in tracks]
    sections = (p.analysis or {}).get("sections") if len(tracks) > 1 else None
    mf = midi.build_midi(specs, bpm, p.time_signature, p.offset_seconds, (p.analysis or {}).get("key"), sections)
    return midi.save_and_validate(mf, out, specs, bpm, p.offset_seconds)


def select_tracks(p: Project, stems: str | None) -> list[Track]:
    if not stems:
        return list(p.tracks)
    wanted = {s.strip() for s in stems.split(",") if s.strip()}
    return [t for t in p.tracks if t.stem in wanted]


# ------------------------------------------------------------------ bundles (export + restore)

BUNDLE_VERSION = 1
_BUNDLE_NAME = re.compile(r"^(stems|midi|artifacts)/[A-Za-z0-9_\-.]{1,120}$|^(project\.json|README\.txt|ATTRIBUTION\.txt|multitrack\.mid)$")
ATTRIBUTION = """Generated with Stemforge.
Source separation: Demucs / HTDemucs (MIT, https://github.com/adefossez/demucs). Weights from
adefossez/HTDemucs* on Hugging Face; no separate weight license is stated - verify before commercial use.
Transcription: Spotify Basic Pitch (Apache-2.0), librosa pYIN (ISC), Stemforge drum DSP.
You are responsible for having the rights to the source recording.
"""


def write_bundle(db: Session, p: Project, out: Path, tracks: list[Track], include_stems: bool, include_midi: bool,
                 full_project: bool) -> None:
    storage = get_storage()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.writestr("ATTRIBUTION.txt", ATTRIBUTION)
        stems = {t.stem for t in tracks}
        if include_stems:
            for a in p.artifacts:
                if a.kind == "stem" and a.stem in stems:
                    with storage.open(a.storage_key) as f:
                        z.writestr(f"stems/{a.filename}", f.read(), compress_type=zipfile.ZIP_STORED)
        if include_midi:
            with tempfile.TemporaryDirectory() as td:
                for t in tracks:
                    if t.transcription_status != "done":
                        continue
                    path = Path(td) / f"{t.stem}.mid"
                    build_export_midi(p, [t], path)
                    z.write(path, f"midi/{_slug(p.name)}-{t.stem}.mid")
                done = [t for t in tracks if t.transcription_status == "done"]
                if done:
                    path = Path(td) / "multitrack.mid"
                    build_export_midi(p, done, path)
                    z.write(path, "multitrack.mid")
        if full_project:
            doc = {
                "bundle_version": BUNDLE_VERSION, "app_version": __version__,
                "project": {
                    "name": p.name, "original_filename": p.original_filename, "source_format": p.source_format,
                    "options": p.options, "analysis": p.analysis, "tempo_bpm": p.tempo_bpm,
                    "time_signature": p.time_signature, "offset_seconds": p.offset_seconds,
                },
                "tracks": [
                    {k: getattr(t, k) for k in ("stem", "display_name", "order_index", "source_model", "is_drum",
                                                "gm_program", "stats", "warnings", "engine", "engine_reason",
                                                "engine_params", "notes", "generated_notes", "notes_version",
                                                "transcription_status", "transcription_detail")}
                    for t in p.tracks
                ],
                "artifacts": [],
            }
            for a in p.artifacts:
                if a.kind in {"master", "preview", "stem", "peaks", "spectrogram", "features"}:
                    name = f"artifacts/{a.kind}-{a.stem or 'mix'}{Path(a.storage_key).suffix}"
                    with storage.open(a.storage_key) as f:
                        z.writestr(name, f.read())
                    doc["artifacts"].append({"path": name, "kind": a.kind, "stem": a.stem, "filename": a.filename,
                                             "content_type": a.content_type, "params_hash": a.params_hash,
                                             "meta": a.meta, "sha256": a.sha256})
            z.writestr("project.json", json.dumps(doc, default=str))


class BundleError(Exception):
    pass


def import_bundle(db: Session, zip_path: Path) -> Project:
    """Restore a full project bundle. Entry names are whitelisted (no traversal) and sizes are capped."""
    settings = get_settings()
    storage = get_storage()
    limit = settings.max_upload_bytes * 12
    try:
        z = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise BundleError("not a valid zip file") from exc
    with z:
        total = 0
        for info in z.infolist():
            if info.is_dir():
                continue
            if not _BUNDLE_NAME.match(info.filename):
                raise BundleError(f"unexpected entry in bundle: {info.filename[:80]!r}")
            total += info.file_size
            if total > limit:
                raise BundleError("bundle too large when extracted")
            if info.compress_size and info.file_size / max(info.compress_size, 1) > 200:
                raise BundleError("suspicious compression ratio")
        try:
            doc = json.loads(z.read("project.json"))
        except KeyError as exc:
            raise BundleError("bundle has no project.json (export with 'Full project' to restore later)") from exc
        if doc.get("bundle_version") != BUNDLE_VERSION:
            raise BundleError("unsupported bundle version")
        meta = doc["project"]
        p = Project(
            name=str(meta.get("name", "Restored project"))[:200], original_filename=str(meta.get("original_filename", ""))[:255],
            source_format=str(meta.get("source_format", "flac"))[:16], options=meta.get("options") or {},
            analysis=meta.get("analysis"), tempo_bpm=meta.get("tempo_bpm"),
            time_signature=meta.get("time_signature", "4/4"), offset_seconds=float(meta.get("offset_seconds") or 0),
            status="ready",
        )
        if settings.retention_days > 0:
            p.expires_at = utcnow() + timedelta(days=settings.retention_days)
        db.add(p)
        db.flush()
        with tempfile.TemporaryDirectory() as td:
            for a in doc.get("artifacts", []):
                path = str(a["path"])
                if not _BUNDLE_NAME.match(path) or not path.startswith("artifacts/"):
                    raise BundleError("bad artifact path")
                local = Path(td) / Path(path).name
                local.write_bytes(z.read(path))
                if a.get("sha256") and sha256_file(local) != a["sha256"]:
                    raise BundleError(f"checksum mismatch for {path}")
                kind, stem = str(a["kind"]), a.get("stem")
                if kind not in {"master", "preview", "stem", "peaks", "spectrogram", "features"}:
                    raise BundleError("bad artifact kind")
                sub = {"master": "", "preview": "preview/", "stem": "stems/", "peaks": "peaks/",
                       "spectrogram": "spectrogram/", "features": "features/"}[kind]
                fname = re.sub(r"[^A-Za-z0-9_\-.]", "_", str(a.get("filename") or local.name))[:120]
                key = f"projects/{p.id}/{sub}{'master.flac' if kind == 'master' else (stem or 'mix') + local.suffix}"
                if kind == "features":
                    key = f"projects/{p.id}/features/{stem}-{(a.get('meta') or {}).get('engine', 'x')}.npz"
                size = storage.put_file(key, local, str(a.get("content_type", "application/octet-stream")))
                db.add(Artifact(project_id=p.id, kind=kind, stem=stem, storage_key=key, filename=fname,
                                content_type=str(a.get("content_type", "application/octet-stream")), size=size,
                                sha256=sha256_file(local), params_hash=str(a.get("params_hash", "")),
                                meta=a.get("meta") or {}))
        for t in doc.get("tracks", []):
            db.add(Track(project_id=p.id, **{k: t[k] for k in (
                "stem", "display_name", "order_index", "source_model", "is_drum", "gm_program", "stats",
                "warnings", "engine", "engine_reason", "engine_params", "notes", "generated_notes",
                "notes_version", "transcription_status", "transcription_detail") if k in t}))
        db.commit()
        db.refresh(p)
        return p


# ------------------------------------------------------------------ retention / deletion


def delete_project(db: Session, p: Project) -> None:
    try:
        get_storage().delete_prefix(f"projects/{p.id}")
    except Exception:  # noqa: BLE001
        log.exception("storage cleanup failed", extra={"project_id": p.id})
    db.delete(p)
    db.commit()


def cleanup_expired(db: Session) -> int:
    now = utcnow()
    expired = db.scalars(select(Project).where(Project.expires_at.is_not(None), Project.expires_at < now)).all()
    for p in expired:
        if latest_job(db, p.id) and latest_job(db, p.id).status in {"queued", "running"}:  # type: ignore[union-attr]
            continue
        log.info("deleting expired project", extra={"project_id": p.id})
        delete_project(db, p)
    # Orphaned temp upload files
    tmp = get_settings().data_dir / "uploads"
    if tmp.is_dir():
        for f in tmp.iterdir():
            if f.is_file() and f.stat().st_mtime < time.time() - 3600:
                f.unlink(missing_ok=True)
    return len(expired)


def copy_stream_to_file(src, dst: Path) -> None:
    with dst.open("wb") as out:
        shutil.copyfileobj(src, out)
