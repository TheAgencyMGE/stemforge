"""Job execution: resumable stage pipeline with real progress, ETA and cancellation."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import queue
from ..config import get_settings
from ..models import Artifact, Job, Project, Track, utcnow
from ..storage import Storage, sha256_file
from . import analysis, audio, midi, transcription
from .separation import DemucsSeparator, residual_db

log = logging.getLogger(__name__)

STAGES = ["decode", "analyze", "separate", "render", "transcribe", "midi"]
STAGE_WEIGHTS = {"decode": 0.04, "analyze": 0.06, "separate": 0.5, "render": 0.12, "transcribe": 0.23, "midi": 0.05}
RETRANSCRIBE_STAGES = ["transcribe", "midi"]
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_LABEL = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "guitar": "Guitar", "piano": "Piano", "other": "Other"}
SILENT_DB = -55.0  # absolute dBFS RMS
RELATIVE_SILENT_DB = -40.0  # dB relative to the mix RMS
LOW_LEVEL_DB = -20.0


class PermanentError(Exception):
    """Errors that retrying cannot fix (bad input, missing model); shown to the user."""


def phash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:32]


class JobContext:
    def __init__(self, db: Session, job: Job, storage: Storage, worker_id: str):
        self.db, self.job, self.storage, self.worker_id = db, job, storage, worker_id
        self.project: Project = db.get(Project, job.project_id)  # type: ignore[assignment]
        self.stage_list = RETRANSCRIBE_STAGES if job.kind == "retranscribe" else STAGES
        weights = {s: STAGE_WEIGHTS[s] for s in self.stage_list}
        total = sum(weights.values())
        self.weights = {s: w / total for s, w in weights.items()}
        self._last_flush = 0.0
        self._stage_t0 = time.monotonic()
        self.tmp = Path(tempfile.mkdtemp(prefix="stemforge-"))

    # -- progress -----------------------------------------------------------
    def begin(self, stage: str, message: str) -> None:
        self._stage_t0 = time.monotonic()
        stages = dict(self.job.stages or {})
        stages[stage] = {"status": "running", "progress": 0.0, "message": message,
                         "started_at": utcnow().isoformat()}
        self.job.stages = stages
        self.job.stage = stage
        self.job.message = message
        self._flush(force=True)

    def progress(self, stage: str, frac: float, message: str | None = None) -> None:
        frac = float(np.clip(frac, 0, 1))
        stages = dict(self.job.stages or {})
        st = dict(stages.get(stage, {}))
        st["progress"] = round(frac, 4)
        if message:
            st["message"] = message
            self.job.message = message
        elapsed = time.monotonic() - self._stage_t0
        # Stage ETA from measured throughput, only once there is enough signal to be meaningful.
        self.job.eta_seconds = round(elapsed / frac * (1 - frac), 1) if 0.1 <= frac < 1 and elapsed > 2 else None
        stages[stage] = st
        self.job.stages = stages
        done = sum(self.weights[s] for s in self.stage_list[: self.stage_list.index(stage)])
        self.job.progress = round(done + self.weights[stage] * frac, 4)
        self._flush()

    def end(self, stage: str, status: str = "done", message: str | None = None, **detail: Any) -> None:
        stages = dict(self.job.stages or {})
        st = dict(stages.get(stage, {}))
        st.update({"status": status, "progress": 1.0, "finished_at": utcnow().isoformat(),
                   "seconds": round(time.monotonic() - self._stage_t0, 2), **detail})
        if message:
            st["message"] = message
        stages[stage] = st
        self.job.stages = stages
        idx = self.stage_list.index(stage)
        self.job.progress = round(sum(self.weights[s] for s in self.stage_list[: idx + 1]), 4)
        self.job.eta_seconds = None
        self._flush(force=True)

    def _flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_flush < 0.5:
            return
        self._last_flush = now
        self.db.commit()
        queue.heartbeat(self.db, self.job, self.worker_id)  # raises JobCancelled

    # -- artifacts ----------------------------------------------------------
    def key(self, *parts: str) -> str:
        return "/".join(["projects", self.project.id, *parts])

    def find_artifact(self, kind: str, stem: str | None, params_hash: str) -> Artifact | None:
        q = select(Artifact).where(Artifact.project_id == self.project.id, Artifact.kind == kind,
                                   Artifact.params_hash == params_hash)
        q = q.where(Artifact.stem.is_(None)) if stem is None else q.where(Artifact.stem == stem)
        art = self.db.scalars(q.order_by(Artifact.created_at.desc())).first()
        if art is not None and self.storage.exists(art.storage_key):
            return art
        return None

    def save_artifact(self, kind: str, stem: str | None, local: Path, key: str, filename: str,
                      content_type: str, params_hash: str, meta: dict | None = None) -> Artifact:
        size = self.storage.put_file(key, local, content_type)
        # Replace older artifacts of the same kind/stem (their storage objects are overwritten or stale).
        q = select(Artifact).where(Artifact.project_id == self.project.id, Artifact.kind == kind)
        q = q.where(Artifact.stem.is_(None)) if stem is None else q.where(Artifact.stem == stem)
        for old in self.db.scalars(q):
            if old.storage_key != key:
                try:
                    self.storage.delete_prefix(old.storage_key)
                except Exception:  # noqa: BLE001 - best effort
                    log.warning("could not delete stale artifact", extra={"key": old.storage_key})
            self.db.delete(old)
        self.db.flush()
        art = Artifact(project_id=self.project.id, kind=kind, stem=stem, storage_key=key, filename=filename,
                       content_type=content_type, size=size, sha256=sha256_file(local),
                       params_hash=params_hash, meta=meta or {})
        self.db.add(art)
        self.db.commit()
        return art

    def read_audio(self, art: Artifact) -> np.ndarray:
        import soundfile as sf

        with self.storage.local_path(art.storage_key) as p:
            data, _ = sf.read(str(p), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T)

    def read_npz(self, art: Artifact) -> dict[str, np.ndarray]:
        with self.storage.open(art.storage_key) as f:
            with np.load(io.BytesIO(f.read()), allow_pickle=False) as z:
                return {k: z[k] for k in z.files}


# ============================================================================ process


def run_job(db: Session, job: Job, storage: Storage, worker_id: str) -> None:
    ctx = JobContext(db, job, storage, worker_id)
    if ctx.project is None:
        raise PermanentError("project no longer exists")
    try:
        if job.kind == "process":
            _run_process(ctx)
        elif job.kind == "retranscribe":
            _run_retranscribe(ctx)
        else:
            raise PermanentError(f"unknown job kind {job.kind}")
    finally:
        import shutil

        shutil.rmtree(ctx.tmp, ignore_errors=True)


def _run_process(ctx: JobContext) -> None:
    p, settings = ctx.project, get_settings()
    p.status = "processing"
    p.error = None
    ctx.db.commit()
    opts = p.options or {}
    source = ctx.db.scalars(select(Artifact).where(Artifact.project_id == p.id, Artifact.kind == "source")).first()
    if source is None:
        raise PermanentError("uploaded source file is missing")

    # ---- decode
    trim = {"start": float(opts.get("trim_start") or 0.0), "end": opts.get("trim_end")}
    h_master = phash({"src": source.sha256, "trim": trim, "sr": audio.TARGET_SR})
    ctx.begin("decode", "Decoding audio")
    master_art = ctx.find_artifact("master", None, h_master)
    if master_art is not None:
        mix = ctx.read_audio(master_art)
        ctx.end("decode", "skipped", "Reused decoded audio from a previous attempt")
    else:
        with ctx.storage.local_path(source.storage_key) as src_path:
            try:
                info = audio.probe(src_path, p.source_format)
                if info.duration > settings.max_duration_seconds + 1:
                    raise PermanentError(f"audio is {info.duration:.0f}s; limit is {settings.max_duration_seconds:.0f}s")
                mix = audio.decode(src_path, p.source_format, trim["start"],
                                   float(trim["end"]) if trim["end"] else None)
            except audio.AudioDecodeError as exc:
                raise PermanentError(str(exc)) from exc
        dur = mix.shape[1] / audio.TARGET_SR
        if dur < settings.min_duration_seconds:
            raise PermanentError(f"audio is too short ({dur:.2f}s)")
        local = ctx.tmp / "master.flac"
        audio.write_flac(local, mix, subtype="PCM_24")
        master_art = ctx.save_artifact("master", None, local, ctx.key("master.flac"), "original.flac",
                                       "audio/flac", h_master, {"duration": dur})
        prev = ctx.tmp / "master_preview.flac"
        audio.write_flac(prev, mix, subtype="PCM_16")
        ctx.save_artifact("preview", "original", prev, ctx.key("preview", "original.flac"), "original.flac",
                          "audio/flac", h_master)
        ctx.end("decode", message=f"Decoded {dur:.1f}s of audio", decoder=info.decoder,
                source_sample_rate=info.sample_rate, source_channels=info.channels)
    p.offset_seconds = trim["start"]
    ctx.db.commit()

    # ---- analyze
    ctx.begin("analyze", "Detecting tempo, beats, key and sections")
    if p.analysis and p.analysis.get("_hash") == h_master:
        ctx.end("analyze", "skipped", "Reused analysis")
    else:
        result = analysis.analyze(mix, audio.TARGET_SR)
        result["_hash"] = h_master
        p.analysis = result
        if p.tempo_bpm is None and result.get("tempo"):
            p.tempo_bpm = result["tempo"]
        ctx.db.commit()
        msg = f"Tempo {result.get('tempo') or '–'} BPM, key {(result.get('key') or {}).get('label', '–')}"
        ctx.end("analyze", message=msg)

    # ---- separate
    six = bool(opts.get("six_stem"))
    base_model = settings.demucs_model
    h_sep = phash({"master": h_master, "model": base_model, "six": six, "shifts": settings.demucs_shifts})
    stems: dict[str, np.ndarray] = {}
    stem_models: dict[str, str] = {}
    ctx.begin("separate", f"Separating with {base_model}" + (" + htdemucs_6s" if six else ""))
    wanted = ["vocals", "drums", "bass", "other"] + (["guitar", "piano"] if six else [])
    existing = {s: ctx.find_artifact("stem", s, h_sep) for s in wanted}
    if all(existing.values()):
        for s, art in existing.items():
            stems[s] = ctx.read_audio(art)  # type: ignore[arg-type]
            stem_models[s] = (art.meta or {}).get("model", base_model)  # type: ignore[union-attr]
        ctx.end("separate", "skipped", "Reused stems from a previous attempt")
    else:
        try:
            sep4 = DemucsSeparator(base_model)
        except Exception as exc:
            raise PermanentError(f"separation model unavailable: {exc}") from exc
        share = 0.6 if six else 1.0
        out4 = sep4.separate(mix, audio.TARGET_SR,
                             lambda f: ctx.progress("separate", f * share, f"{base_model}: {f * 100:.0f}%"))
        for s in ["vocals", "drums", "bass", "other"]:
            stems[s], stem_models[s] = out4[s], sep4.name
        if six:
            try:
                sep6 = DemucsSeparator("htdemucs_6s")
            except Exception as exc:
                raise PermanentError(f"6-stem model unavailable: {exc}") from exc
            out6 = sep6.separate(mix, audio.TARGET_SR,
                                 lambda f: ctx.progress("separate", 0.6 + 0.4 * f, f"htdemucs_6s: {f * 100:.0f}%"))
            # Vocals/drums/bass from the stronger 4-stem model; guitar, piano and the remaining
            # "other" from the 6-stem model so the stems still (approximately) sum to the mix.
            for s in ["guitar", "piano", "other"]:
                stems[s], stem_models[s] = out6[s], sep6.name
        for s in wanted:
            local = ctx.tmp / f"{s}.flac"
            audio.write_flac(local, stems[s], subtype="PCM_24")
            ctx.save_artifact("stem", s, local, ctx.key("stems", f"{s}.flac"), f"{_slug(p.name)}-{s}.flac",
                              "audio/flac", h_sep, {"model": stem_models[s]})
        ctx.end("separate", message=f"{len(wanted)} stems", residual_db=residual_db(mix, stems))
    resid = residual_db(mix, stems)

    # ---- tracks + render (previews, peaks, spectrograms)
    _upsert_tracks(ctx, stems, stem_models, resid, audio.level_stats(mix)["rms_db"])
    ctx.begin("render", "Rendering waveforms and spectrograms")
    renders = [("original", mix)] + [(s, stems[s]) for s in _ordered(stems)]
    for i, (name, data) in enumerate(renders):
        h_r = phash({"sep": h_sep if name != "original" else h_master, "stem": name, "v": 2})
        if not ctx.find_artifact("peaks", name, h_r):
            local = ctx.tmp / f"{name}.peaks.json"
            local.write_text(json.dumps(audio.compute_peaks(data, audio.TARGET_SR)))
            ctx.save_artifact("peaks", name, local, ctx.key("peaks", f"{name}.json"), f"{name}.peaks.json",
                              "application/json", h_r)
        if name != "original" and not ctx.find_artifact("preview", name, h_r):
            local = ctx.tmp / f"{name}.preview.flac"
            audio.write_flac(local, data, subtype="PCM_16")
            ctx.save_artifact("preview", name, local, ctx.key("preview", f"{name}.flac"), f"{name}.flac",
                              "audio/flac", h_r)
        if name != "original" and name != "drums" and not ctx.find_artifact("spectrogram", name, h_r):
            local = ctx.tmp / f"{name}.cqt.png"
            meta = audio.pitch_spectrogram_png(data, audio.TARGET_SR, local)
            ctx.save_artifact("spectrogram", name, local, ctx.key("spectrogram", f"{name}.png"),
                              f"{name}.png", "image/png", h_r, meta)
        ctx.progress("render", (i + 1) / len(renders))
    ctx.end("render")

    # ---- transcribe
    tracks = sorted(ctx.db.scalars(select(Track).where(Track.project_id == p.id)).all(),
                    key=lambda t: t.order_index)
    ctx.begin("transcribe", "Transcribing stems to MIDI")
    for i, tr in enumerate(tracks):
        if tr.engine is None:
            eng, params, reason = transcription.choose_engine(tr.stem)
            tr.engine, tr.engine_params, tr.engine_reason = eng, params, reason
        label = f"Transcribing {tr.display_name} ({tr.engine})"
        _transcribe_track(ctx, tr, stems[tr.stem], h_sep, preserve_edits=True,
                          frac=lambda f, i=i, label=label: ctx.progress("transcribe", (i + f) / len(tracks), label))
    ctx.end("transcribe")

    # ---- midi
    ctx.begin("midi", "Writing and validating MIDI")
    _write_midi(ctx)
    ctx.end("midi")
    p.status = "ready"
    settings_ret = settings.retention_days
    p.expires_at = utcnow() + timedelta(days=settings_ret) if settings_ret > 0 and not p.is_sample else None
    ctx.db.commit()


def _ordered(stems: dict[str, Any]) -> list[str]:
    return [s for s in STEM_ORDER if s in stems]


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:60] or "project"


def _upsert_tracks(ctx: JobContext, stems: dict[str, np.ndarray], models: dict[str, str], resid: float,
                   mix_rms_db: float) -> None:
    from ..model_registry import manifest

    existing = {t.stem: t for t in ctx.db.scalars(select(Track).where(Track.project_id == ctx.project.id))}
    for idx, s in enumerate(_ordered(stems)):
        stats = audio.level_stats(stems[s])
        stats["relative_db"] = round(stats["rms_db"] - mix_rms_db, 2)
        stats["silent"] = bool(stats["rms_db"] < SILENT_DB or stats["relative_db"] < RELATIVE_SILENT_DB)
        stats["mix_residual_db"] = resid
        warnings = []
        if "6s" in models[s] and s == "piano":
            warnings.append(manifest()["htdemucs_6s"]["caveat"])
        if s == "other":
            warnings.append("Residual stem: may contain several instruments; transcription can be ambiguous.")
        if stats["silent"]:
            warnings.append(f"Stem is effectively silent ({stats['rms_db']} dBFS RMS, "
                            f"{stats['relative_db']} dB below the mix); transcription skipped.")
        elif stats["relative_db"] < LOW_LEVEL_DB:
            warnings.append(f"Stem is very quiet ({stats['relative_db']} dB relative to the mix); notes may "
                            "come from bleed of other instruments rather than a real part.")
        if stats["clipped_ratio"] > 0.001:
            warnings.append("Stem has clipped samples; separation output exceeded full scale.")
        tr = existing.get(s)
        # Edits only survive a re-run if the stem audio came from the same model.
        stats["model_changed"] = bool(tr is not None and tr.source_model and tr.source_model != models[s])
        tr = tr or Track(project_id=ctx.project.id, stem=s)
        if tr not in ctx.db:
            ctx.db.add(tr)
        tr.display_name = tr.display_name or STEM_LABEL.get(s, s.title())
        tr.order_index = idx
        tr.source_model = models[s]
        tr.is_drum = s == "drums"
        tr.gm_program = transcription.GM_PROGRAM.get(s, 0)
        tr.stats = stats
        tr.warnings = warnings
    ctx.db.commit()


def _transcribe_track(ctx: JobContext, tr: Track, data: np.ndarray, h_sep: str, preserve_edits: bool,
                      frac) -> None:
    if (tr.stats or {}).get("silent"):
        tr.transcription_status = "skipped"
        tr.transcription_detail = "Stem is silent; nothing to transcribe."
        tr.generated_notes, tr.notes = [], []
        tr.notes_version += 1
        ctx.db.commit()
        frac(1.0)
        return
    engine = tr.engine or transcription.choose_engine(tr.stem)[0]
    params = transcription.resolve_params(engine, tr.engine_params)
    h_feat = phash({"sep": h_sep, "stem": tr.stem, "engine": engine,
                    "fp": transcription.feature_key_params(engine, params), "v": 1})
    art = ctx.find_artifact("features", tr.stem, h_feat)
    if art is not None:
        feats = ctx.read_npz(art)
    else:
        feats = transcription.compute_features(engine, data, audio.TARGET_SR, params)
        local = ctx.tmp / f"{tr.stem}.{engine}.npz"
        np.savez_compressed(local, **feats)
        ctx.save_artifact("features", tr.stem, local, ctx.key("features", f"{tr.stem}-{engine}.npz"),
                          f"{tr.stem}-{engine}.npz", "application/octet-stream", h_feat, {"engine": engine})
    frac(0.7)
    notes = transcription.notes_from_features(engine, feats, data, audio.TARGET_SR, params)
    edited = tr.notes_version > 0 and tr.notes != tr.generated_notes and not (tr.stats or {}).get("model_changed")
    tr.generated_notes = notes
    if not (preserve_edits and edited):
        tr.notes = notes
        tr.notes_version += 1
    tr.engine, tr.engine_params = engine, params
    tr.transcription_status = "done"
    summary = transcription.summarize(notes)
    tr.transcription_detail = json.dumps(summary)
    ctx.db.commit()
    frac(1.0)


def _write_midi(ctx: JobContext) -> None:
    p = ctx.project
    tracks = sorted(ctx.db.scalars(select(Track).where(Track.project_id == p.id)).all(), key=lambda t: t.order_index)
    bpm = p.tempo_bpm or (p.analysis or {}).get("tempo") or 120.0
    key = (p.analysis or {}).get("key")
    sections = (p.analysis or {}).get("sections") or []
    with_notes = [t for t in tracks if t.transcription_status == "done"]
    for i, t in enumerate(with_notes):
        spec = [midi.MidiTrackSpec(t.display_name, t.notes, t.gm_program, t.is_drum)]
        mf = midi.build_midi(spec, bpm, p.time_signature, p.offset_seconds, key)
        local = ctx.tmp / f"{t.stem}.mid"
        report = midi.save_and_validate(mf, local, spec, bpm, p.offset_seconds)
        ctx.save_artifact("midi", t.stem, local, ctx.key("midi", f"{t.stem}.mid"),
                          f"{_slug(p.name)}-{t.stem}.mid", "audio/midi", phash(t.notes_version), report)
        ctx.progress("midi", (i + 1) / (len(with_notes) + 1))
    specs = [midi.MidiTrackSpec(t.display_name, t.notes, t.gm_program, t.is_drum) for t in with_notes]
    mf = midi.build_midi(specs, bpm, p.time_signature, p.offset_seconds, key, sections)
    local = ctx.tmp / "combined.mid"
    report = midi.save_and_validate(mf, local, specs, bpm, p.offset_seconds)
    ctx.save_artifact("midi_combined", None, local, ctx.key("midi", "combined.mid"),
                      f"{_slug(p.name)}-multitrack.mid", "audio/midi", "", report)


# ============================================================================ retranscribe


def _run_retranscribe(ctx: JobContext) -> None:
    params = ctx.job.params or {}
    tr = ctx.db.get(Track, params.get("track_id"))
    if tr is None or tr.project_id != ctx.project.id:
        raise PermanentError("track not found")
    engine = params.get("engine") or tr.engine
    if engine not in transcription.ENGINES:
        raise PermanentError(f"unknown engine {engine}")
    if engine == "drum-nmf" and not tr.is_drum:
        raise PermanentError("the drum engine only applies to the drum stem")
    stem_art = ctx.db.scalars(select(Artifact).where(Artifact.project_id == ctx.project.id,
                                                     Artifact.kind == "stem", Artifact.stem == tr.stem)).first()
    if stem_art is None:
        raise PermanentError("stem audio missing; re-run processing")
    data = ctx.read_audio(stem_art)
    tr.engine = engine
    tr.engine_params = transcription.resolve_params(engine, params.get("params"))
    tr.engine_reason = "Selected manually" if params.get("engine") else tr.engine_reason
    ctx.db.commit()
    ctx.begin("transcribe", f"Re-transcribing {tr.display_name} with {engine}")
    # A re-transcription replaces edits on this track: that is what the user asked for.
    _transcribe_track(ctx, tr, data, stem_art.params_hash, preserve_edits=False,
                      frac=lambda f: ctx.progress("transcribe", f))
    ctx.end("transcribe")
    ctx.begin("midi", "Writing and validating MIDI")
    _write_midi(ctx)
    ctx.end("midi")
