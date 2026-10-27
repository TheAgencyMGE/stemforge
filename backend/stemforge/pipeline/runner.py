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
from .discovery import DISCOVERY_VERSION, discover
from .drums import piece_audio
from .instruments import get_instrument, stem_blurb
from .refine import CROSS_SOURCE_RATIO, comparison_engine, cross_source_duplicates, engine_disagreements
from .refine import apply as refine_notes
from .separation import DemucsSeparator, gate_mixture_silence

log = logging.getLogger(__name__)

STAGES = ["decode", "analyze", "separate", "render", "transcribe", "midi"]
STAGE_WEIGHTS = {"decode": 0.04, "analyze": 0.06, "separate": 0.5, "render": 0.12, "transcribe": 0.23, "midi": 0.05}
RETRANSCRIBE_STAGES = ["transcribe", "midi"]
STEM_ORDER = [
    "vocals", "drums", "kit_kick", "kit_snare", "kit_hats", "kit_perc",
    "bass", "guitar", "piano", "other", "mix",
]
STEM_LABEL = {
    "vocals": "Vocals", "drums": "Drums", "bass": "Bass", "guitar": "Guitar", "piano": "Piano",
    "other": "Other", "mix": "Full mix",
    "kit_kick": "Kick (mask)", "kit_snare": "Snare (mask)", "kit_hats": "Hats (mask)",
    "kit_perc": "Percussion (mask)",
}
KIT_PIECES = ["kit_kick", "kit_snare", "kit_hats", "kit_perc"]
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

    # ---- separate + discover which sources are really there
    stems, stem_models, source_meta, h_sep, resid, reused_stems = _separate_and_discover(ctx, mix, h_master, opts)
    _maybe_drum_kit(ctx, stems, stem_models, h_sep, opts)
    if opts.get("transcribe_mix"):
        stems["mix"] = mix
        stem_models["mix"] = "basic-pitch:full-mix"
    shown = [s for s in source_meta]
    if reused_stems:
        ctx.end("separate", "skipped", "Reused stems from a previous attempt", residual_db=resid)
    else:
        ctx.end("separate", message=f"{len(shown)} sources found", residual_db=resid)

    # ---- tracks + render (previews, peaks, spectrograms)
    _upsert_tracks(ctx, stems, stem_models, resid, audio.level_stats(mix)["rms_db"], source_meta)
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
        pitched = name != "original" and name != "drums" and not name.startswith("kit_")
        if pitched and not ctx.find_artifact("spectrogram", name, h_r):
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
        if tr.stem.startswith("kit_"):
            tr.engine = tr.engine or "drum-nmf"
            tr.engine_params = tr.engine_params or {}
            tr.engine_reason = "Listening mask of the drum stem. Hits are transcribed on the Drums track."
        elif tr.engine is None:
            eng, params, reason = transcription.choose_engine(tr.stem)
            identity = ((tr.stats or {}).get("source") or {}).get("identity")
            inst = get_instrument(identity) if identity else None
            if eng == "onset-events" and inst is not None and inst.gm_note is not None:
                params = {**params, "gm_note": inst.gm_note}
            if inst is not None:
                reason = f"{reason} Source identified as {inst.name}."
            tr.engine, tr.engine_params, tr.engine_reason = eng, params, reason
        label = f"Transcribing {tr.display_name} ({tr.engine})"
        _transcribe_track(ctx, tr, stems[tr.stem], h_sep, preserve_edits=True,
                          frac=lambda f, i=i, label=label: ctx.progress("transcribe", (i + f) / len(tracks), label))
    _attribute_duplicates(ctx)
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
    """Known stems in their usual order, then any discovered source ids alphabetically."""
    known = [s for s in STEM_ORDER if s in stems]
    return known + sorted(s for s in stems if s not in STEM_ORDER)


def _separate_and_discover(ctx: JobContext, mix: np.ndarray, h_master: str, opts: dict[str, Any]):
    """Broad separation, then evidence-based discovery of the sources that are actually present.

    Returns (stems, stem_models, source_meta, h_sep, mixture residual dB, reused). Only sources the
    evidence supports are returned; rejected stems never become tracks (see discovery.py).
    """
    from . import tagger

    settings = get_settings()
    base_model = settings.demucs_model
    six = bool(opts.get("six_stem"))
    h_sep = phash({"master": h_master, "model": base_model, "six": six, "shifts": settings.demucs_shifts,
                   "discovery": DISCOVERY_VERSION, "tagger": tagger.available(), "gate": 1})
    ctx.begin("separate", f"Separating with {base_model} and listening for instruments")
    disc_art = ctx.find_artifact("discovery", None, h_sep)
    if disc_art is not None:
        with ctx.storage.open(disc_art.storage_key) as f:
            summary = json.loads(f.read())
        found = {src["stem"]: ctx.find_artifact("stem", src["stem"], h_sep) for src in summary["sources"]}
        if all(found.values()):
            cached = {name: ctx.read_audio(art) for name, art in found.items()}  # type: ignore[arg-type]
            cached_models = {name: (art.meta or {}).get("model", base_model)  # type: ignore[union-attr]
                             for name, art in found.items()}
            _store_discovery(ctx, summary)
            return cached, cached_models, {src["stem"]: src for src in summary["sources"]}, h_sep, summary["reconstruction"]["residual_db"], True

    separators: dict[str, DemucsSeparator] = {}

    def run(model: str, audio_in: np.ndarray, cb) -> dict[str, np.ndarray]:
        try:
            if model not in separators:
                separators[model] = DemucsSeparator(model)
        except Exception as exc:
            if model == base_model:
                raise PermanentError(f"separation model unavailable: {exc}") from exc
            raise
        out = separators[model].separate(audio_in, audio.TARGET_SR, cb)
        return gate_mixture_silence(audio_in, out, audio.TARGET_SR)

    base = run(base_model, mix, lambda f: ctx.progress("separate", 0.5 * f, f"{base_model}: {f * 100:.0f}%"))
    result = discover(
        mix, audio.TARGET_SR, base, base_model, run,
        lambda f, msg: ctx.progress("separate", 0.5 + 0.48 * f, msg), opts,
    )
    summary = result.summary()
    stems: dict[str, np.ndarray] = {}
    models: dict[str, str] = {}
    meta: dict[str, dict[str, Any]] = {}
    for src_meta in summary["sources"]:
        name = src_meta["stem"]
        stems[name] = result.sources[name].audio
        models[name] = src_meta.get("separation") or base_model
        meta[name] = src_meta
        local = ctx.tmp / f"{name}.flac"
        flac = audio.write_flac_verified(local, stems[name], subtype="PCM_24")
        ctx.save_artifact("stem", name, local, ctx.key("stems", f"{name}.flac"),
                          f"{_slug(ctx.project.name)}-{name}.flac", "audio/flac", h_sep,
                          {"model": models[name], "status": src_meta["status"], "flac": flac})
    if result.internal_residual is not None:
        # Kept for reconstruction checks and explicit download; never a track.
        local = ctx.tmp / "internal_residual.flac"
        audio.write_flac(local, result.internal_residual, subtype="PCM_24")
        ctx.save_artifact("residual", None, local, ctx.key("stems", "internal_residual.flac"),
                          f"{_slug(ctx.project.name)}-internal-residual.flac", "audio/flac", h_sep,
                          {"rel_db": summary["reconstruction"].get("internal_residual_db")})
    _delete_stale_stems(ctx, set(stems))
    local = ctx.tmp / "discovery.json"
    local.write_text(json.dumps(summary, default=float))
    ctx.save_artifact("discovery", None, local, ctx.key("discovery.json"), "discovery.json",
                      "application/json", h_sep)
    _store_discovery(ctx, summary)
    return stems, models, meta, h_sep, summary["reconstruction"]["residual_db"], False


def _store_discovery(ctx: JobContext, summary: dict[str, Any]) -> None:
    analysis_doc = dict(ctx.project.analysis or {})
    analysis_doc["discovery"] = {k: summary[k] for k in ("version", "tagger", "log", "reconstruction")}
    ctx.project.analysis = analysis_doc
    ctx.db.commit()


def _delete_stale_stems(ctx: JobContext, keep: set[str]) -> None:
    """Remove artifacts of sources a previous run produced but this run does not support."""
    kinds = ["stem", "preview", "peaks", "spectrogram", "midi", "features"]
    q = select(Artifact).where(Artifact.project_id == ctx.project.id, Artifact.kind.in_(kinds))
    for art in ctx.db.scalars(q):
        stem = art.stem or ""
        base = stem.split("__", 1)[0]
        if not stem or stem == "original" or stem.startswith("kit_") or stem == "mix" or base in keep:
            continue
        try:
            ctx.storage.delete_prefix(art.storage_key)
        except Exception:  # noqa: BLE001 - best effort
            log.warning("could not delete stale artifact", extra={"key": art.storage_key})
        ctx.db.delete(art)
    ctx.db.commit()


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:60] or "project"


def _upsert_tracks(ctx: JobContext, stems: dict[str, np.ndarray], models: dict[str, str], resid: float,
                   mix_rms_db: float, source_meta: dict[str, dict[str, Any]] | None = None) -> None:
    from ..model_registry import manifest

    source_meta = source_meta or {}

    existing = {t.stem: t for t in ctx.db.scalars(select(Track).where(Track.project_id == ctx.project.id))}
    used_names: dict[str, int] = {}
    for idx, s in enumerate(_ordered(stems)):
        stats = audio.level_stats(stems[s])
        stats["relative_db"] = round(stats["rms_db"] - mix_rms_db, 2)
        stats["silent"] = bool(stats["rms_db"] < SILENT_DB or stats["relative_db"] < RELATIVE_SILENT_DB)
        stats["mix_residual_db"] = resid
        warnings = []
        if "6s" in models[s] and s == "piano":
            warnings.append(manifest()["htdemucs_6s"]["caveat"])
        meta = source_meta.get(s)
        if meta is not None:
            # Discovery already established that this source is audible; the old global-RMS
            # silence rule would only re-litigate that with less evidence.
            stats["silent"] = False
            stats["source"] = {k: v for k, v in meta.items() if not k.startswith("_")}
            if meta.get("status") == "unresolved":
                cands = ", ".join(c["name"] for c in meta.get("candidates", [])[:3]) or "no candidate above noise"
                warnings.append(f"Unresolved source: audible content that no tag identifies with confidence "
                                f"(closest: {cands}). Kept for review and for a complete reconstruction.")
            elif meta.get("isolation") == "group":
                warnings.append("Several instruments are audible in this track and no bundled model separates "
                                "them from each other: " + ", ".join(c["name"] for c in meta.get("contains", [])))
            for absorbed in meta.get("absorbed", []):
                warnings.append(f"Includes bleed reassigned from the {absorbed['from']} stem, which had no "
                                f"part of its own ({absorbed['rel_db']} dB).")
            warnings.extend(meta.get("limitations", [])[:2])
        if stats["silent"]:
            warnings.append(f"Stem is effectively silent ({stats['rms_db']} dBFS RMS, "
                            f"{stats['relative_db']} dB below the mix); transcription skipped.")
        elif stats["relative_db"] < LOW_LEVEL_DB:
            warnings.append(f"Stem is very quiet ({stats['relative_db']} dB relative to the mix); notes may "
                            "come from bleed of other instruments rather than a real part.")
        if stats["clipped_ratio"] > 0.001:
            warnings.append("Stem has clipped samples; separation output exceeded full scale.")
        blurb = stem_blurb(s)
        # The generic residual blurb ("instruments here are not identified") is only true when
        # discovery could not name the source.
        if blurb and not (meta is not None and s == "other" and meta.get("status") == "accepted"
                          and meta.get("kind") != "unresolved"):
            warnings.insert(0, blurb)
        tr = existing.get(s)
        # Edits only survive a re-run if the stem audio came from the same model.
        stats["model_changed"] = bool(tr is not None and tr.source_model and tr.source_model != models[s])
        tr = tr or Track(project_id=ctx.project.id, stem=s)
        if tr not in ctx.db:
            ctx.db.add(tr)
        inst = get_instrument(meta["identity"]) if meta and meta.get("identity") else None
        if inst is None and meta and meta.get("contains"):
            # A group or unresolved source: use its strongest identified member for the GM program.
            inst = get_instrument(meta["contains"][0]["id"])
        if meta is not None:
            name = meta.get("display") or STEM_LABEL.get(s, s.title())
            used_names[name] = used_names.get(name, 0) + 1
            tr.display_name = name if used_names[name] == 1 else f"{name} {used_names[name]}"
        else:
            tr.display_name = tr.display_name or STEM_LABEL.get(s, s.title())
        tr.order_index = idx
        tr.source_model = models[s]
        tr.is_drum = s in {"drums", "percussion"} or s.startswith("kit_")
        if inst is not None and inst.gm_program is not None:
            tr.gm_program = inst.gm_program
        else:
            tr.gm_program = transcription.GM_PROGRAM.get(s, 0)
        tr.stats = stats
        tr.warnings = warnings
    for stem, old in existing.items():
        # A source this run did not find is not kept as an empty track.
        if stem not in stems:
            ctx.db.delete(old)
    ctx.db.commit()


def _maybe_drum_kit(ctx: JobContext, stems: dict[str, np.ndarray], models: dict[str, str], h_sep: str,
                    opts: dict[str, Any]) -> None:
    """Optional NMF masks of the drum stem. They are not added to the mix-residual sum."""
    if not opts.get("drum_kit") or "drums" not in stems:
        return
    h_kit = phash({"sep": h_sep, "kit": "nmf-v1"})
    found = {name: ctx.find_artifact("stem", name, h_kit) for name in KIT_PIECES}
    if all(found.values()):
        for name, art in found.items():
            stems[name] = ctx.read_audio(art)  # type: ignore[arg-type]
            models[name] = (art.meta or {}).get("model", "nmf-mask:drum-stem")  # type: ignore[union-attr]
        return
    ctx.progress("separate", 0.98, "Splitting the drum stem into kit pieces")
    for name, data in piece_audio(stems["drums"], audio.TARGET_SR).items():
        stems[name] = data
        models[name] = "nmf-mask:drum-stem"
        local = ctx.tmp / f"{name}.flac"
        audio.write_flac(local, data, subtype="PCM_24")
        ctx.save_artifact("stem", name, local, ctx.key("stems", f"{name}.flac"),
                          f"{_slug(ctx.project.name)}-{name}.flac", "audio/flac", h_kit,
                          {"model": models[name], "parent": "drums"})


def _transcribe_track(ctx: JobContext, tr: Track, data: np.ndarray, h_sep: str, preserve_edits: bool,
                      frac) -> None:
    if tr.stem.startswith("kit_"):
        tr.transcription_status = "skipped"
        tr.transcription_detail = json.dumps({
            "count": 0,
            "detail": "Kit piece is a mask for listening. Drum hits are on the Drums track, so this piece is not transcribed again.",
        })
        tr.generated_notes, tr.notes = [], []
        tr.notes_version += 1
        ctx.db.commit()
        frac(1.0)
        return
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
    if engine == "muscriptor" and not params.get("instruments"):
        # Condition MuScriptor on what discovery identified, the model's own documented option.
        from .external import REGISTRY_TO_MUSCRIPTOR

        identity = ((tr.stats or {}).get("source") or {}).get("identity")
        groups = REGISTRY_TO_MUSCRIPTOR.get(identity or "", [])
        if tr.stem == "drums":
            groups = ["drums"]
        params["instruments"] = ",".join(groups)
    notes = _notes_for_engine(ctx, tr.stem, engine, params, data, h_sep, tr.stem)
    # Raw model output is kept next to the refined notes so every correction stays reviewable.
    local = ctx.tmp / f"{tr.stem}.raw.json"
    local.write_text(json.dumps({"engine": engine, "params": params, "notes": notes}))
    ctx.save_artifact("notes_raw", tr.stem, local, ctx.key("notes", f"{tr.stem}-raw.json"),
                      f"{_slug(ctx.project.name)}-{tr.stem}-raw-notes.json", "application/json", h_sep,
                      {"engine": engine, "count": len(notes)})
    frac(0.7)
    refined = refine_notes(tr.stem, engine, notes, data, audio.TARGET_SR)
    suggestions = list(refined.suggestions)
    if (ctx.project.options or {}).get("compare_engines"):
        secondary = comparison_engine(tr.stem, engine)
        if secondary:
            band = {k: params[k] for k in ("min_freq", "max_freq") if params.get(k) is not None}
            sec_params = transcription.resolve_params(secondary, band)
            # A separate artifact stem so the comparison cache does not replace the primary one.
            sec_notes = _notes_for_engine(ctx, tr.stem, secondary, sec_params, data, h_sep, f"{tr.stem}__{secondary}")
            suggestions.extend(engine_disagreements(refined.notes, sec_notes, secondary))
    edited = tr.notes_version > 0 and tr.notes != tr.generated_notes and not (tr.stats or {}).get("model_changed")
    tr.generated_notes = refined.notes
    if not (preserve_edits and edited):
        tr.notes = refined.notes
        tr.notes_version += 1
    tr.engine, tr.engine_params = engine, params
    tr.transcription_status = "done"
    summary = transcription.summarize(refined.notes)
    summary["suggestions"] = suggestions
    summary["coverage"] = stem_blurb(tr.stem)
    if refined.tablature is not None:
        summary["tablature"] = refined.tablature
    tr.transcription_detail = json.dumps(summary)
    ctx.db.commit()
    frac(1.0)


def _mix_attributed_notes(ctx: JobContext, stem: str, h_sep: str, params: dict[str, Any]) -> list[dict]:
    """YourMT3+ on the whole mix (once per separation), each note assigned to the pitched source that
    plays it (attribution.py), with velocity from that source's loudness. Cached on the context."""
    import librosa

    from .attribution import attribute_notes

    cache = getattr(ctx, "_mix_attribution", None)
    if cache is None or cache[0] != h_sep:
        master = ctx.db.scalars(select(Artifact).where(Artifact.project_id == ctx.project.id,
                                                       Artifact.kind == "master")).first()
        if master is None:
            raise PermanentError("decoded audio missing; re-run processing")
        mix = ctx.read_audio(master)
        mix_notes = _notes_for_engine(ctx, "mix", "yourmt3", {**params, "input": "stem"}, mix, h_sep, "mix__yourmt3")
        sources: dict[str, np.ndarray] = {}
        for t in ctx.db.scalars(select(Track).where(Track.project_id == ctx.project.id)):
            if t.is_drum or t.stem == "mix" or t.stem.startswith("kit_"):
                continue
            art = ctx.find_artifact("stem", t.stem, h_sep)
            if art is not None:
                sources[t.stem] = ctx.read_audio(art)
        assigned = attribute_notes(mix_notes, sources, audio.TARGET_SR)
        for name, notes in assigned.items():
            y = librosa.resample(audio.to_mono(sources[name]), orig_sr=audio.TARGET_SR, target_sr=22050)
            curve = transcription._velocity_curve(y, 22050)
            for n in notes:
                n["velocity"] = curve(float(n["start"]), float(n["end"]))
        cache = (h_sep, assigned)
        ctx._mix_attribution = cache  # type: ignore[attr-defined]
    return [dict(n) for n in cache[1].get(stem, [])]


def _attribute_duplicates(ctx: JobContext) -> None:
    """Remove bleed copies: a note that a clearly louder pitched source also has (refine.py).

    Only unedited tracks change, the removed notes are listed in the track's review notes, and the
    raw model notes stay in the notes_raw artifact.
    """
    tracks = [t for t in ctx.db.scalars(select(Track).where(Track.project_id == ctx.project.id))
              if t.transcription_status == "done" and not t.is_drum and t.stem != "mix"]
    if len(tracks) < 2:
        return
    dupes = cross_source_duplicates({t.stem: list(t.generated_notes or []) for t in tracks})
    for t in tracks:
        ids = set(dupes.get(t.stem, []))
        if not ids:
            continue
        unedited = (t.notes or []) == (t.generated_notes or [])
        removed = [n for n in t.generated_notes or [] if n["id"] in ids]
        t.generated_notes = [n for n in t.generated_notes or [] if n["id"] not in ids]
        if unedited:
            t.notes = list(t.generated_notes)
            t.notes_version += 1
        detail = json.loads(t.transcription_detail or "{}") if (t.transcription_detail or "").startswith("{") else {}
        others = sorted({o.display_name for o in tracks if o is not t})
        detail.setdefault("suggestions", []).append({
            "code": "cross_source_bleed",
            "severity": "info",
            "message": (f"{len(removed)} notes were bleed copies of notes that {', '.join(others)} plays at least "
                        f"{CROSS_SOURCE_RATIO:.1f}x louder at the same pitch and onset, and were attributed there. "
                        "The raw model notes are kept with the track."),
            "note_ids": [],
        })
        detail["count"] = len(t.generated_notes)
        t.transcription_detail = json.dumps(detail)
    ctx.db.commit()


def _notes_for_engine(ctx: JobContext, stem: str, engine: str, params: dict[str, Any], data: np.ndarray,
                      h_sep: str, feature_stem: str) -> list[dict]:
    if engine == "yourmt3" and params.get("input") == "mix":
        return _mix_attributed_notes(ctx, stem, h_sep, params)
    h_feat = phash({"sep": h_sep, "stem": stem, "engine": engine,
                    "fp": transcription.feature_key_params(engine, params), "v": 1})
    art = ctx.find_artifact("features", feature_stem, h_feat)
    if art is not None:
        feats = ctx.read_npz(art)
    else:
        feats = transcription.compute_features(engine, data, audio.TARGET_SR, params)
        local = ctx.tmp / f"{feature_stem}.{engine}.npz"
        np.savez_compressed(local, **feats)
        ctx.save_artifact("features", feature_stem, local, ctx.key("features", f"{feature_stem}-{engine}.npz"),
                          f"{feature_stem}-{engine}.npz", "application/octet-stream", h_feat, {"engine": engine})
    return transcription.notes_from_features(engine, feats, data, audio.TARGET_SR, params)


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
    _render_fidelity(ctx, with_notes, bpm)


def _render_fidelity(ctx: JobContext, tracks: list[Track], bpm: float) -> None:
    """Render each track's MIDI through the configured General MIDI soundfont and measure it against
    the track's own stem (render.compare). Skipped when no renderer is configured."""
    from . import render

    if render.renderer() is None:
        return
    for t in tracks:
        stem_art = ctx.find_artifact("stem", t.stem, _latest_hash(ctx, "stem", t.stem))
        if stem_art is None or not t.notes:
            continue
        stem_audio = ctx.read_audio(stem_art)
        local = ctx.tmp / f"{t.stem}.render.mid"
        spec = [midi.MidiTrackSpec(t.display_name, t.notes, t.gm_program, t.is_drum)]
        midi.build_midi(spec, bpm).save(str(local))  # processed-region timeline, like the stem
        try:
            rendered = render.render_midi(local, audio.TARGET_SR, stem_audio.shape[1] / audio.TARGET_SR)
        except render.RenderError as exc:
            log.warning("MIDI render failed", extra={"stem": t.stem, "error": str(exc)})
            continue
        out = ctx.tmp / f"{t.stem}.render.flac"
        audio.write_flac(out, rendered, subtype="PCM_16")
        ctx.save_artifact("midi_render", t.stem, out, ctx.key("render", f"{t.stem}.flac"),
                          f"{_slug(ctx.project.name)}-{t.stem}-midi-render.flac", "audio/flac", phash(t.notes_version),
                          {"soundfont": Path(get_settings().soundfont_path).name, "gm_program": t.gm_program})
        t.stats = {**(t.stats or {}), "render_fidelity": render.compare(rendered, stem_audio, audio.TARGET_SR)}
    ctx.db.commit()


def _latest_hash(ctx: JobContext, kind: str, stem: str) -> str:
    art = ctx.db.scalars(select(Artifact).where(Artifact.project_id == ctx.project.id, Artifact.kind == kind,
                                                Artifact.stem == stem).order_by(Artifact.created_at.desc())).first()
    return art.params_hash if art is not None else ""


# ============================================================================ retranscribe


def _run_retranscribe(ctx: JobContext) -> None:
    params = ctx.job.params or {}
    tr = ctx.db.get(Track, params.get("track_id"))
    if tr is None or tr.project_id != ctx.project.id:
        raise PermanentError("track not found")
    engine = params.get("engine") or tr.engine
    if engine not in transcription.allowed_engines(tr.stem, tr.is_drum):
        raise PermanentError(f"engine {engine} is not available for this track")
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
    _attribute_duplicates(ctx)
    ctx.end("transcribe")
    ctx.begin("midi", "Writing and validating MIDI")
    _write_midi(ctx)
    ctx.end("midi")
