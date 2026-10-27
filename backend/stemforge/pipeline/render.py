"""Render MIDI through a General MIDI soundfont, and measure how close audio signals are.

Rendering uses the FluidSynth program (LGPL, run as a separate process) and any SF2/SF3
soundfont; the evaluation uses MuseScore General (MIT, derived from FluidR3). Configure with
``STEMFORGE_FLUIDSYNTH_PATH`` and ``STEMFORGE_SOUNDFONT_PATH``. Nothing here is needed for the core pipeline.

The metrics separate *what* is played from *how it sounds*:

* ``chroma_similarity``: mean cosine between frame chroma vectors in frames where either signal
  is active. Pitch classes and harmony; largely independent of instrument timbre.
* ``onset_correlation``: correlation of onset-strength envelopes. Rhythm and timing.
* ``logmel_l1_db``: mean absolute difference of level-normalised log-mel spectra. Everything,
  including timbre, effects and mixing.

Comparing render(predicted MIDI) with render(reference MIDI) through the same soundfont isolates
note errors. Comparing render(reference MIDI) with the real audio measures what an instrument
choice cannot recover (samples, effects, mix). See docs/EVALUATION.md.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..config import get_settings
from .audio import to_mono

SR = 22050
HOP = 512


class RenderError(Exception):
    pass


def renderer() -> tuple[Path, Path] | None:
    s = get_settings()
    if not s.fluidsynth_path or not s.soundfont_path:
        return None
    fs, sf2 = Path(s.fluidsynth_path), Path(s.soundfont_path)
    return (fs, sf2) if fs.is_file() and sf2.is_file() else None


def render_midi(midi_path: Path, sr: int = 44100, seconds: float | None = None) -> np.ndarray:
    """(2, n) float32 audio of ``midi_path`` through the configured soundfont, starting at t=0."""
    found = renderer()
    if found is None:
        raise RenderError("Set STEMFORGE_FLUIDSYNTH_PATH and STEMFORGE_SOUNDFONT_PATH to render MIDI.")
    fs, sf2 = found
    with tempfile.TemporaryDirectory(prefix="sf-render-") as td:
        wav = Path(td) / "out.wav"
        # -n no MIDI input, -i non-interactive, -q quiet, -F file output, -r rate, -g gain (fixed so
        # renders are comparable), reverb and chorus off so only the notes and samples differ.
        proc = subprocess.run(
            [str(fs), "-niq", "-R", "0", "-C", "0", "-g", "0.6", "-r", str(sr), "-F", str(wav), str(sf2),
             str(midi_path)], capture_output=True, text=True, timeout=1200, check=False)
        if proc.returncode != 0 or not wav.is_file():
            raise RenderError(f"fluidsynth failed: {(proc.stderr or proc.stdout)[-400:]}")
        data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    out = data.T
    if rate != sr:
        import soxr

        out = soxr.resample(out.T, rate, sr).T
    if seconds is not None:
        n = int(seconds * sr)
        out = out[:, :n] if out.shape[1] >= n else np.pad(out, ((0, 0), (0, n - out.shape[1])))
    return np.ascontiguousarray(out, dtype=np.float32)


def _features(x: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    import librosa

    y = librosa.resample(to_mono(x).astype(np.float32), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    rms = float(np.sqrt(np.mean(y ** 2))) or 1e-9
    y = y / rms  # level-normalised: overall gain is not a fidelity error
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=96), ref=1.0,
                              amin=1e-8)
    return {
        "mel": mel,
        "chroma": librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP),
        "onset": librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP),
        "active": librosa.feature.rms(y=y * rms, hop_length=HOP)[0] > 1e-3,
    }


def compare(a: np.ndarray, b: np.ndarray, sr: int) -> dict[str, Any]:
    """Distances between two signals on the same timeline (see module docstring)."""
    fa, fb = _features(a, sr), _features(b, sr)
    n = min(fa["mel"].shape[1], fb["mel"].shape[1])
    active = (fa["active"][:n] | fb["active"][:n])
    if active.sum() < 4:
        return {"chroma_similarity": None, "onset_correlation": None, "logmel_l1_db": None, "frames": int(active.sum())}
    ca, cb = fa["chroma"][:, :n][:, active], fb["chroma"][:, :n][:, active]
    cos = (ca * cb).sum(0) / (np.linalg.norm(ca, axis=0) * np.linalg.norm(cb, axis=0) + 1e-9)
    oa, ob = fa["onset"][:n], fb["onset"][:n]
    corr = float(np.corrcoef(oa, ob)[0, 1]) if oa.std() > 0 and ob.std() > 0 else 0.0
    ma = np.maximum(fa["mel"][:, :n][:, active], -80.0)
    mb = np.maximum(fb["mel"][:, :n][:, active], -80.0)
    return {"chroma_similarity": round(float(cos.mean()), 3), "onset_correlation": round(corr, 3),
            "logmel_l1_db": round(float(np.abs(ma - mb).mean()), 2), "frames": int(active.sum())}
