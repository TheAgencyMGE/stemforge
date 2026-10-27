"""Audio export: which tracks are exported, in which timeline mode, and a manifest that proves it.

Two modes, because they answer different needs:

* ``synced``: every exported stem is the full project length and starts at the same instant,
  so dropping the files into a DAW at 0:00 reproduces the arrangement. Nothing is trimmed.
* ``trimmed``: standalone files. Leading and trailing silence are removed (never internal
  silence), and the manifest records where each file starts on the original timeline.

Tracks with no audible content are excluded and listed with the reason. Muting is a mixer state:
it decides what goes into a rendered mix, never what happens to a stored stem.

The manifest separates four properties that are easy to conflate (see ``PROPERTIES``).
"""

from __future__ import annotations

import hashlib
import io
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from .models import Artifact, Project, Track
from .pipeline import audio as audio_mod
from .pipeline.activity import silence_bounds
from .storage import Storage

Mode = Literal["synced", "trimmed"]

PROPERTIES = {
    "lossless_encoding": "Each FLAC decodes to exactly the 24-bit PCM Stemforge wrote (checked on export).",
    "separation_accuracy": "A stem is a model's estimate of one source, not the studio multitrack. Bleed and "
                           "artifacts are possible and are not measured by the lossless check.",
    "mix_reconstruction": "How well the stems sum back to the original is reported separately "
                          "(reconstruction.residual_db): the sum of all sources minus the mix, relative to the mix.",
    "midi_fidelity": "MIDI is a transcription of the stem and is validated for structure and timing, "
                     "not for musical correctness.",
}


def audio_artifact(p: Project, t: Track) -> Artifact | None:
    return next((a for a in p.artifacts if a.kind == "stem" and a.stem == t.stem), None)


def is_exportable_by_default(t: Track) -> tuple[bool, str | None]:
    """Default selection: real, audible sources. Unresolved sources are included (they are real audio
    and needed to rebuild the mix) and marked as such in the manifest. Everything else is opt-in."""
    stats = t.stats or {}
    if stats.get("silent"):
        return False, "no audible content (reported silent)"
    if t.stem.startswith("kit_"):
        return False, "listening mask derived from the drum stem (duplicates drum audio)"
    if t.stem == "mix":
        return False, "transcription of the original mix; it has no separate audio"
    return True, None


def default_tracks(p: Project) -> list[Track]:
    return [t for t in p.tracks if is_exportable_by_default(t)[0]]


def _read_pcm(storage: Storage, art: Artifact) -> tuple[np.ndarray, int, int]:
    """Stored stem as integer PCM (channels, n), its rate and bit depth. Nothing is re-quantized."""
    with storage.open(art.storage_key) as f:
        data = f.read()
    info = sf.info(io.BytesIO(data))
    bits = {"PCM_16": 16, "PCM_24": 24}.get(info.subtype, 24)
    pcm, sr = sf.read(io.BytesIO(data), dtype="int32", always_2d=True)
    return (pcm >> (32 - bits)).T, int(sr), bits


def _write_pcm(path: Path, pcm: np.ndarray, sr: int, bits: int) -> None:
    subtype = "PCM_24" if bits == 24 else "PCM_16"
    sf.write(str(path), np.ascontiguousarray((pcm.astype(np.int64) << (32 - bits)).astype(np.int32).T), sr,
             format="FLAC", subtype=subtype)
    back, _ = sf.read(str(path), dtype="int32", always_2d=True)
    if not np.array_equal(back.T >> (32 - bits), pcm):
        raise audio_mod.FlacVerificationError(f"{path.name}: decoded PCM differs from the exported samples")


def _sha(pcm: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pcm.T.astype(np.int32)).tobytes()).hexdigest()


def export_track_audio(storage: Storage, p: Project, t: Track, mode: Mode, out_dir: Path,
                       trim_threshold_db: float = -60.0) -> tuple[Path | None, dict[str, Any]]:
    """Write one track's audio for export. Returns (file or None, manifest entry)."""
    art = audio_artifact(p, t)
    source = (t.stats or {}).get("source") or {}
    entry: dict[str, Any] = {"track": t.display_name, "stem": t.stem, "identity": source.get("identity"),
                             "status": source.get("status", "accepted")}
    if art is None:
        return None, {**entry, "excluded": "no audio for this track"}
    pcm, sr, bits = _read_pcm(storage, art)
    n = pcm.shape[1]
    offset = float(p.offset_seconds or 0.0)
    if mode == "trimmed":
        bounds = silence_bounds(pcm.astype(np.float64) / float(2 ** (bits - 1)), sr, threshold_db=trim_threshold_db)
        if bounds.empty:
            return None, {**entry, "excluded": f"no sample above {trim_threshold_db} dBFS"}
        start, end = bounds.start_sample, bounds.end_sample
    else:
        start, end = 0, n
    piece = pcm[:, start:end]
    out = out_dir / f"{art.filename.rsplit('.', 1)[0]}{'-trimmed' if mode == 'trimmed' and (start or end < n) else ''}.flac"
    _write_pcm(out, piece, sr, bits)
    entry.update({
        "file": out.name, "mode": mode, "format": "FLAC", "sample_rate": sr, "channels": int(piece.shape[0]),
        "bit_depth": bits, "frames": int(piece.shape[1]),
        "timeline_start_seconds": round(offset + start / sr, 6),
        "timeline_end_seconds": round(offset + end / sr, 6),
        "trimmed_leading_seconds": round(start / sr, 6), "trimmed_trailing_seconds": round((n - end) / sr, 6),
        "pcm_sha256": _sha(piece), "lossless_check": "decoded PCM == stored stem samples",
    })
    return out, entry


def render_mix(storage: Storage, p: Project, tracks: list[Track], out: Path) -> dict[str, Any]:
    """Sum of the given tracks at unity gain on the full timeline. No normalization: if the sum
    exceeds full scale the clipped sample count is reported instead of silently rescaling."""
    total: np.ndarray | None = None
    used: list[str] = []
    sr = audio_mod.TARGET_SR
    for t in tracks:
        art = audio_artifact(p, t)
        if art is None:
            continue
        pcm, sr, bits = _read_pcm(storage, art)
        x = pcm.astype(np.float64) / float(2 ** (bits - 1))
        total = x if total is None else total + x[:, : total.shape[1]]
        used.append(t.display_name)
    if total is None:
        raise ValueError("no audio tracks to mix")
    info = audio_mod.write_flac_verified(out, total.astype(np.float32), sr, "PCM_24")
    return {"file": out.name, "tracks": used, "gain": "unity", "normalized": False,
            "timeline_start_seconds": float(p.offset_seconds or 0.0), **info}


def manifest_header(p: Project, mode: Mode) -> dict[str, Any]:
    disc = (p.analysis or {}).get("discovery") or {}
    return {
        "project": p.name, "mode": mode, "original_file": p.original_filename,
        "timeline": {"origin": "start of the original file",
                     "processed_offset_seconds": float(p.offset_seconds or 0.0)},
        "reconstruction": disc.get("reconstruction"),
        "properties": PROPERTIES, "files": [], "excluded": [], "midi": [],
    }


def scratch_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="sf-audio-"))


# ----------------------------------------------------------------------------- Soundtrap package

SOUNDTRAP_AUDIO_FORMATS = ("mp3", "wav", "aif", "mp4", "m4a", "ogg", "aac")  # Soundtrap help centre, Oct 2026


def soundtrap_bpm(p: Project) -> int:
    """Soundtrap has one project tempo, typed in BPM. Writing the MIDI at an integer tempo lets the
    user type it exactly; note times are converted at that tempo, so they stay exact in seconds."""
    bpm = p.tempo_bpm or (p.analysis or {}).get("tempo") or 120.0
    return int(round(float(bpm)))


def _write_wav_pcm(path: Path, pcm: np.ndarray, sr: int, bits: int) -> None:
    subtype = "PCM_24" if bits == 24 else "PCM_16"
    sf.write(str(path), np.ascontiguousarray((pcm.astype(np.int64) << (32 - bits)).astype(np.int32).T), sr,
             format="WAV", subtype=subtype)
    back, _ = sf.read(str(path), dtype="int32", always_2d=True)
    if not np.array_equal(back.T >> (32 - bits), pcm):
        raise audio_mod.FlacVerificationError(f"{path.name}: decoded PCM differs from the stem samples")


def soundtrap_package(storage: Storage, p: Project, tracks: list[Track], out_dir: Path,
                      build_midi) -> dict[str, Any]:
    """Files for rebuilding the song in Soundtrap: per-track MIDI at one integer tempo, full-length
    24-bit WAV stems (Soundtrap does not import FLAC), and a manifest. Returns the manifest."""
    from .pipeline.instruments import get_instrument

    bpm = soundtrap_bpm(p)
    offset = float(p.offset_seconds or 0.0)
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(tracks, start=1):
        base = f"{i:02d} {re.sub(r'[^A-Za-z0-9 +_-]', '', t.display_name).strip() or t.stem}"
        row: dict[str, Any] = {"track": t.display_name, "stem": t.stem, "is_drum": t.is_drum,
                               "gm_program": None if t.is_drum else t.gm_program}
        source = (t.stats or {}).get("source") or {}
        inst = get_instrument(source["identity"]) if source.get("identity") else None
        row["instrument_family"] = inst.family if inst else source.get("family")
        row["gm_name"] = "Drum kit (GM channel 10)" if t.is_drum else _gm_name(t.gm_program)
        if t.transcription_status == "done" and t.notes:
            midi_path = out_dir / f"{base}.mid"
            build_midi(p, [t], midi_path, bpm)
            row["midi"] = midi_path.name
            row["notes"] = len(t.notes)
        art = audio_artifact(p, t)
        if art is not None and not (t.stats or {}).get("silent"):
            pcm, sr, bits = _read_pcm(storage, art)
            wav_path = out_dir / f"{base}.wav"
            _write_wav_pcm(wav_path, pcm, sr, bits)
            row.update({"audio": wav_path.name, "sample_rate": sr, "bit_depth": bits, "frames": int(pcm.shape[1]),
                        "timeline_start_seconds": offset, "pcm_sha256": _sha(pcm)})
        fidelity = (t.stats or {}).get("render_fidelity")
        if fidelity:
            row["render_fidelity"] = fidelity
        rows.append(row)
    return {"bpm": bpm, "detected_bpm": p.tempo_bpm or (p.analysis or {}).get("tempo"),
            "time_signature": p.time_signature, "offset_seconds": offset, "tracks": rows}


def _gm_name(program: int | None) -> str | None:
    if program is None:
        return None
    import pretty_midi

    return f"{pretty_midi.program_to_instrument_name(int(program))} (GM {int(program) + 1})"


def soundtrap_readme(man: dict[str, Any]) -> str:
    lines = [
        "# Rebuilding this song in Soundtrap",
        "",
        f"1. Create a new Soundtrap project and set its tempo to **{man['bpm']} BPM** "
        f"(time signature {man['time_signature']}) *before* importing anything.",
        "   Soundtrap places MIDI in beats and audio in time. The MIDI files here are written at exactly "
        f"{man['bpm']} BPM, so their notes line up with the audio only when the project uses that tempo.",
        "2. Drag the `.wav` stems onto the timeline and move each region to 0:00. They are all full length "
        "and start together, so they line up with each other and with the original.",
        "3. Drag each `.mid` file onto the timeline at bar 1, beat 1, and pick an instrument for it "
        "(suggestions below). Each MIDI file holds one track.",
        "4. Mute a WAV stem and solo its MIDI track to compare what the transcription recovered.",
        "",
        "| # | Track | Notes | Suggested instrument | Audio stem |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(man["tracks"], start=1):
        fam = r.get("instrument_family") or ""
        sugg = r.get("gm_name") or ""
        if fam:
            sugg = f"{sugg}; any {fam} instrument in the Soundtrap picker" if sugg else f"a {fam} instrument"
        lines.append(f"| {i} | {r['track']} | {r.get('notes', '-')} | {sugg} | {r.get('audio', '-')} |")
    lines += [
        "",
        "## What the MIDI cannot carry",
        "",
        "* Timbre, effects (reverb, distortion, compression), mixing and mastering. Choosing a Soundtrap "
        "instrument gives an approximation of the part, not the original sound. Keep the WAV stems for that.",
        "* Pitch bends are written only where a part plays one note at a time; MIDI bends apply to a whole channel.",
        "* Tempo changes: Soundtrap has one project tempo. Notes keep their real timing, so a song that "
        "speeds up or slows down will not sit on the bar lines, but it will line up with the audio.",
        "* Drums are written on General MIDI channel 10 with GM drum notes. How Soundtrap maps them depends "
        "on the drum instrument you choose; this was not tested in Soundtrap itself.",
        "",
        "Generated by Stemforge. Separated stems are model estimates; transcriptions are checked for "
        "structure and timing, not musical correctness. See export_manifest.json for exact numbers.",
    ]
    return "\n".join(lines) + "\n"
