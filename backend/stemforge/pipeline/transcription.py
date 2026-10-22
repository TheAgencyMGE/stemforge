"""Per-stem transcription engines and the explainable engine-selection policy.

Every engine returns notes as dicts:
    {"id", "pitch", "start", "end", "velocity", "confidence", "bends"}
Times are seconds relative to the processed audio's t=0. `confidence` is derived from the
engine's own probabilities (never invented); `bends` is a list of [t_rel_seconds, semitones]
or None.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .audio import to_mono, write_wav

log = logging.getLogger(__name__)
SR = 22050

GM_DRUMS = {"kick": 36, "snare": 38, "hihat": 42}


@dataclass
class EngineSpec:
    id: str
    label: str
    polyphonic: bool
    description: str
    defaults: dict[str, Any] = field(default_factory=dict)


ENGINES: dict[str, EngineSpec] = {
    "basic-pitch": EngineSpec(
        "basic-pitch", "Basic Pitch (polyphonic)", True,
        "Spotify Basic Pitch neural AMT via ONNX Runtime. Polyphonic, instrument-agnostic, with pitch bends.",
        {"onset_threshold": 0.5, "frame_threshold": 0.3, "min_note_ms": 80, "min_freq": None, "max_freq": None},
    ),
    "pyin": EngineSpec(
        "pyin", "pYIN melody (monophonic)", False,
        "Probabilistic YIN f0 tracking with HMM smoothing, segmented into notes; pitch bends from the f0 contour.",
        {"onset_threshold": 0.5, "frame_threshold": 0.5, "min_note_ms": 90, "min_freq": 65.0, "max_freq": 1100.0},
    ),
    "drum-nmf": EngineSpec(
        "drum-nmf", "Drum NMF (kick/snare/hi-hat)", True,
        "Semi-adaptive NMF of the drum stem into kick, snare and hi-hat templates; onsets are peak-picked "
        "per template and mapped to GM 36/38/42.",
        {"onset_threshold": 0.5, "min_note_ms": 40},
    ),
}

STEM_POLICY: dict[str, tuple[str, dict[str, Any], str]] = {
    "vocals": ("pyin", {"min_freq": 70.0, "max_freq": 1100.0},
               "Vocals are predominantly monophonic; pYIN gives calibrated voicing probabilities and "
               "smooth pitch contours (vibrato becomes pitch bend rather than extra notes)."),
    "bass": ("basic-pitch", {"min_freq": 30.0, "max_freq": 500.0, "onset_threshold": 0.6},
             "Band-limited (30-500 Hz) Basic Pitch. On the evaluation fixture it beat pYIN (note F1 0.89 vs "
             "0.79) because it re-articulates repeated notes; switch to pYIN for legato lines."),
    "drums": ("drum-nmf", {},
              "Unpitched percussion needs event detection, not pitch tracking. NMF separates simultaneous "
              "kick/snare/hi-hat hits that a single onset detector would merge."),
    "piano": ("basic-pitch", {"frame_threshold": 0.3},
              "Piano is polyphonic; Basic Pitch transcribes chords. Note: 6-stem piano separation bleeds."),
    "guitar": ("basic-pitch", {"frame_threshold": 0.3, "min_freq": 75.0},
               "Guitar parts are often chordal; Basic Pitch handles polyphony and string bends."),
    "other": ("basic-pitch", {},
              "The residual stem mixes several instruments, so a polyphonic, instrument-agnostic model is used."),
}

GM_PROGRAM = {"vocals": 53, "bass": 33, "piano": 0, "guitar": 25, "other": 48, "drums": 0}


def choose_engine(stem: str) -> tuple[str, dict[str, Any], str]:
    engine, overrides, reason = STEM_POLICY.get(stem, STEM_POLICY["other"])
    params = {**ENGINES[engine].defaults, **overrides}
    return engine, params, reason


def resolve_params(engine: str, params: dict[str, Any] | None) -> dict[str, Any]:
    spec = ENGINES[engine]
    merged = dict(spec.defaults)
    for k, v in (params or {}).items():
        if k in spec.defaults and (v is not None or k in ("min_freq", "max_freq")):
            merged[k] = v
    for k in ("onset_threshold", "frame_threshold"):
        if k in merged:
            merged[k] = float(np.clip(float(merged[k]), 0.05, 0.95))
    merged["min_note_ms"] = float(np.clip(float(merged["min_note_ms"]), 10, 2000))
    for k in ("min_freq", "max_freq"):
        if merged.get(k) is not None:
            merged[k] = float(np.clip(float(merged[k]), 20.0, 8000.0))
    if merged.get("min_freq") and merged.get("max_freq") and merged["min_freq"] >= merged["max_freq"]:
        merged["max_freq"] = None
    return merged


def _nid() -> str:
    return uuid.uuid4().hex[:12]


def _velocity_curve(y: np.ndarray, sr: int, hop: int = 256):
    """Return a function mapping time -> MIDI velocity from the stem's own loudness at that time."""
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    top = float(np.percentile(db, 99.5)) if db.size else 0.0

    def at(t0: float, t1: float) -> int:
        a = int(t0 * sr / hop)
        b = max(a + 1, int(min(t1, t0 + 0.08) * sr / hop))
        seg = db[a:b]
        level = float(seg.max()) if seg.size else -90.0
        return int(np.clip(round(30 + 97 * (level - (top - 36)) / 36), 1, 127))

    return at


# ---------------------------------------------------------------- Basic Pitch


@lru_cache
def _bp_model():
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    return Model(ICASSP_2022_MODEL_PATH)


def basic_pitch_infer(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Run the network once; thresholds are applied afterwards so re-tuning needs no inference."""
    from basic_pitch.inference import run_inference

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "stem.wav"
        write_wav(p, to_mono(audio), sr, subtype="FLOAT")
        out = run_inference(p, _bp_model())
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def basic_pitch_notes(model_output: dict[str, np.ndarray], y: np.ndarray, params: dict[str, Any]) -> list[dict]:
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.note_creation import model_output_to_notes

    min_frames = int(np.round(params["min_note_ms"] / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    _, events = model_output_to_notes(
        model_output, onset_thresh=params["onset_threshold"], frame_thresh=params["frame_threshold"],
        infer_onsets=True, min_note_len=max(1, min_frames), min_freq=params.get("min_freq"),
        max_freq=params.get("max_freq"), include_pitch_bends=True, multiple_pitch_bends=False,
        melodia_trick=True,
    )
    vel = _velocity_curve(y, SR)
    events = sorted(events, key=lambda e: (e[0], e[2]))
    notes = []
    for i, (start, end, pitch, amp, bends) in enumerate(events):
        start, end = float(start), float(end)
        overlapped = any(
            o is not events[i] and o[0] < end and o[1] > start for o in events[max(0, i - 8): i + 8]
        )
        notes.append({
            "id": _nid(), "pitch": int(pitch), "start": round(start, 4), "end": round(end, 4),
            "velocity": vel(start, end), "confidence": round(float(np.clip(amp, 0, 1)), 3),
            # Pitch bend is per MIDI channel, so only monophonic passages keep bends.
            "bends": None if overlapped else _bp_bends(bends, start, end),
        })
    return notes


def _bp_bends(bends: list[int] | None, start: float, end: float) -> list[list[float]] | None:
    if not bends or max(abs(b) for b in bends) < 1:
        return None
    from basic_pitch.constants import CONTOURS_BINS_PER_SEMITONE

    times = np.linspace(0, end - start, len(bends))
    out: list[list[float]] = []
    last = None
    for t, b in zip(times, bends, strict=True):
        semis = round(b / CONTOURS_BINS_PER_SEMITONE, 3)
        if semis != last:
            out.append([round(float(t), 4), semis])
            last = semis
    if out and out[-1][1] != 0:
        out.append([round(end - start, 4), 0.0])
    return out


# ---------------------------------------------------------------- pYIN


def pyin_track(y: np.ndarray, params: dict[str, Any]) -> dict[str, np.ndarray]:
    import librosa

    fmin = float(params.get("min_freq") or 65.0)
    fmax = float(params.get("max_freq") or 1100.0)
    frame_length = 4096 if fmin < 60 else 2048
    hop = 256
    f0, voiced, prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=SR, frame_length=frame_length, hop_length=hop, fill_na=np.nan,
    )
    return {"f0": f0.astype(np.float32), "voiced_prob": prob.astype(np.float32), "hop": np.array([hop])}


def pyin_notes(track: dict[str, np.ndarray], y: np.ndarray, params: dict[str, Any]) -> list[dict]:
    import librosa

    f0, prob = track["f0"], track["voiced_prob"]
    hop = int(track["hop"][0])
    fps = SR / hop
    voicing_thr = params["frame_threshold"]
    midi = librosa.hz_to_midi(np.where(np.isfinite(f0), f0, 1.0))
    voiced = np.isfinite(f0) & (prob >= voicing_thr)

    # Re-articulation points (same pitch, new attack) from the stem's onset envelope.
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    delta = 0.05 + 0.5 * params["onset_threshold"]
    onsets = set(librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=hop, delta=delta,
                                            normalize=True).tolist())
    min_frames = max(2, int(params["min_note_ms"] / 1000 * fps))
    vel = _velocity_curve(y, SR, hop)

    notes: list[dict] = []
    i, n = 0, len(f0)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i + 1
        ref = midi[i]
        while j < n and voiced[j]:
            window = midi[max(i, j - 6): j + 1]
            if abs(midi[j] - np.median(window[:-1] if len(window) > 1 else window)) > 0.7 and abs(midi[j] - ref) > 0.7:
                break  # pitch jump -> new note
            if j in onsets and j - i >= min_frames:
                break  # re-attack on the same pitch
            j += 1
        if j - i >= min_frames:
            seg = midi[i:j]
            pitch = int(np.round(np.median(seg)))
            start, end = i / fps, j / fps
            dev = seg - pitch
            bends = None
            if np.max(np.abs(dev)) > 0.3:
                k = max(1, int(0.02 * fps))  # one bend point per ~20 ms
                pts = [[round(float(t / fps), 4), round(float(np.clip(d, -2, 2)), 3)]
                       for t, d in zip(range(0, len(dev), k), dev[::k], strict=True)]
                pts.append([round(end - start, 4), 0.0])
                bends = pts
            notes.append({
                "id": _nid(), "pitch": pitch, "start": round(start, 4), "end": round(end, 4),
                "velocity": vel(start, end), "confidence": round(float(np.mean(prob[i:j])), 3),
                "bends": bends,
            })
        i = j
    return notes


# ---------------------------------------------------------------- Drums
#
# Semi-adaptive NMF drum transcription (after Dittmar & Gärtner, "Real-time transcription and
# separation of drum recordings based on NMF decomposition", DAFx 2014): the drum stem's magnitude
# spectrogram is factorised with three spectral templates (kick, snare, hi-hat) that start from
# fixed priors and are allowed to adapt halfway towards the recording; onsets are peak-picked on
# each template's activation novelty.

DRUM_CLASSES = ["kick", "snare", "hihat"]


def _drum_templates(freqs: np.ndarray) -> np.ndarray:
    lf = np.log2(np.maximum(freqs, 20.0))

    def bump(center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((lf - np.log2(center)) / width) ** 2)

    kick = bump(60, 0.6)
    snare = 0.6 * bump(200, 0.4) + 0.5 * ((freqs > 1000) & (freqs < 9000))
    hihat = 1.0 * (freqs > 6000) + 0.05
    W = np.stack([kick, snare, hihat], axis=1) + 1e-6
    return W / W.sum(axis=0, keepdims=True)


def drum_features(audio: np.ndarray, sr: int, iters: int = 40, fixed_iters: int = 10,
                  adapt: float = 0.5) -> dict[str, np.ndarray]:
    import librosa

    y = to_mono(audio)
    n_fft, hop = 2048, 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)).astype(np.float64)
    W0 = _drum_templates(librosa.fft_frequencies(sr=sr, n_fft=n_fft))
    W = W0.copy()
    H = np.maximum(W.T @ S, 1e-6)
    for it in range(iters):  # KL-divergence multiplicative updates
        H *= (W.T @ (S / (W @ H + 1e-9))) / (W.sum(axis=0)[:, None] + 1e-9)
        if it >= fixed_iters:
            Wn = W * ((S / (W @ H + 1e-9)) @ H.T) / (H.sum(axis=1)[None, :] + 1e-9)
            Wn /= Wn.sum(axis=0, keepdims=True)
            W = adapt * W0 + (1 - adapt) * Wn
    return {"H": H.astype(np.float32), "fps": np.array([sr / hop], dtype=np.float64)}


def drum_notes(feats: dict[str, np.ndarray], params: dict[str, Any]) -> list[dict]:
    import librosa

    H, fps = feats["H"], float(feats["fps"][0])
    delta = 0.02 + 0.3 * params["onset_threshold"]
    wait = max(1, int(params["min_note_ms"] / 1000 * fps))
    notes = []
    for i, name in enumerate(DRUM_CLASSES):
        h = H[i] / (float(np.percentile(H[i], 99)) + 1e-9)
        nov = np.maximum(0.0, np.diff(h, prepend=h[:1]))
        peaks = librosa.util.peak_pick(nov, pre_max=3, post_max=3, pre_avg=20, post_avg=5, delta=delta, wait=wait)
        for p in peaks:
            local = float(np.mean(nov[max(0, p - 20): p + 6]))
            thr = local + delta
            # Confidence: how far the onset rises above the adaptive threshold, as a fraction of its height.
            conf = float(np.clip((nov[p] - thr) / max(float(nov[p]), 1e-9) * 1.5, 0.0, 1.0))
            level = float(h[p: p + 4].max()) if p < len(h) else float(h[p - 1])
            t = p / fps
            notes.append({
                "id": _nid(), "pitch": GM_DRUMS[name], "start": round(t, 4), "end": round(t + 0.06, 4),
                "velocity": int(np.clip(round(35 + 92 * min(level, 1.0)), 1, 127)),
                "confidence": round(conf, 3), "bends": None,
            })
    notes.sort(key=lambda x: (x["start"], x["pitch"]))
    return notes


# ---------------------------------------------------------------- facade


def compute_features(engine: str, audio: np.ndarray, sr: int, params: dict[str, Any]) -> dict[str, np.ndarray]:
    """The expensive, threshold-independent part of each engine (cached as an artifact)."""
    import librosa

    if engine == "basic-pitch":
        return basic_pitch_infer(audio, sr)
    if engine == "pyin":
        y = librosa.resample(to_mono(audio), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
        return pyin_track(y, params)
    if engine == "drum-nmf":
        return drum_features(audio, sr)
    raise ValueError(f"unknown engine {engine}")


def notes_from_features(engine: str, feats: dict[str, np.ndarray], audio: np.ndarray, sr: int,
                        params: dict[str, Any]) -> list[dict]:
    import librosa

    if engine == "drum-nmf":
        return drum_notes(feats, params)
    y = librosa.resample(to_mono(audio), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    if engine == "basic-pitch":
        return basic_pitch_notes(feats, y, params)
    if engine == "pyin":
        return pyin_notes(feats, y, params)
    raise ValueError(f"unknown engine {engine}")


def feature_key_params(engine: str, params: dict[str, Any]) -> dict[str, Any]:
    """Parameters that change the cached features (others only change post-processing)."""
    if engine == "pyin":
        return {"min_freq": params.get("min_freq"), "max_freq": params.get("max_freq")}
    return {}


def summarize(notes: list[dict], low_conf: float = 0.4) -> dict[str, Any]:
    if not notes:
        return {"count": 0, "mean_confidence": None, "low_confidence": 0}
    conf = np.array([n["confidence"] for n in notes])
    return {
        "count": len(notes), "mean_confidence": round(float(conf.mean()), 3),
        "low_confidence": int((conf < low_conf).sum()),
        "pitch_range": [int(min(n["pitch"] for n in notes)), int(max(n["pitch"] for n in notes))],
    }
