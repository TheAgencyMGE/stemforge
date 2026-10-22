"""Drum-hit refinement and kit-piece masks on top of the three-class NMF.

The factorisation is the same semi-adaptive NMF the transcriber already uses (kick, snare,
hi-hat templates). Relabeling and the listening masks do not add a learned model. ADTOF is
not used: its weights are noncommercial.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .audio import to_mono

N_FFT = 2048
HOP = 256
CLASSES = ("kick", "snare", "hihat")


def _templates(freqs: np.ndarray) -> np.ndarray:
    lf = np.log2(np.maximum(freqs, 20.0))

    def bump(center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((lf - np.log2(center)) / width) ** 2)

    kick = bump(60, 0.6)
    snare = 0.6 * bump(200, 0.4) + 0.5 * ((freqs > 1000) & (freqs < 9000))
    hihat = 1.0 * (freqs > 6000) + 0.05
    w = np.stack([kick, snare, hihat], axis=1) + 1e-6
    return w / w.sum(axis=0, keepdims=True)


def factorize(audio: np.ndarray, sr: int, iters: int = 40, fixed_iters: int = 10,
              adapt: float = 0.5) -> dict[str, Any]:
    """KL-NMF of a mono magnitude spectrogram. ``H`` matches the cached drum features."""
    import librosa

    y = to_mono(audio)
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)).astype(np.float64)
    w0 = _templates(librosa.fft_frequencies(sr=sr, n_fft=N_FFT))
    w = w0.copy()
    h = np.maximum(w.T @ spec, 1e-6)
    for it in range(iters):
        h *= (w.T @ (spec / (w @ h + 1e-9))) / (w.sum(axis=0)[:, None] + 1e-9)
        if it >= fixed_iters:
            wn = w * ((spec / (w @ h + 1e-9)) @ h.T) / (h.sum(axis=1)[None, :] + 1e-9)
            wn /= wn.sum(axis=0, keepdims=True)
            w = adapt * w0 + (1 - adapt) * wn
    return {"W": w, "H": h, "S": spec, "fps": sr / HOP}


def piece_audio(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Wiener masks for kick, snare, hats and whatever the three templates left over.

    Every piece has the drum stem's channel count, sample rate and length. Masks come from
    the mono mix of the stem and are applied per channel, so nothing is time-shifted.
    """
    import librosa

    fac = factorize(audio, sr)
    w, h, spec = fac["W"], fac["H"], fac["S"]
    contrib = [(w[:, i:i + 1] @ h[i:i + 1, :]) for i in range(3)]
    explained = contrib[0] + contrib[1] + contrib[2]
    denom = spec + 1e-9
    masks = {
        "kit_kick": contrib[0] / denom,
        "kit_snare": contrib[1] / denom,
        "kit_hats": contrib[2] / denom,
        "kit_perc": np.maximum(spec - explained, 0.0) / denom,
    }
    stfts = [librosa.stft(audio[c], n_fft=N_FFT, hop_length=HOP) for c in range(audio.shape[0])]
    n = audio.shape[1]
    out: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        channels = []
        for spec_c in stfts:
            frames = min(mask.shape[1], spec_c.shape[1])
            bins = min(mask.shape[0], spec_c.shape[0])
            shaped = spec_c.copy()
            shaped[:bins, :frames] *= mask[:bins, :frames].astype(spec_c.dtype)
            if frames < spec_c.shape[1]:
                shaped[:, frames:] = 0
            channels.append(librosa.istft(shaped, hop_length=HOP, length=n))
        out[name] = np.stack(channels).astype(np.float32)
    return out


def refine_drum_notes(notes: list[dict], audio: np.ndarray, sr: int, params: dict[str, Any]) -> list[dict]:
    """Relabel coarse hits whose spectrum disagrees with the template, then add residual toms.

    Kick, snare and closed-hat hits that match their templates keep GM 36/38/42. Notes are
    not deleted.
    """
    y = to_mono(audio)
    refined = [_relabel(dict(n), y, sr) for n in notes]
    refined.extend(_residual_toms(y, sr, refined, params))
    refined.sort(key=lambda n: (n["start"], n["pitch"]))
    return refined


def _relabel(note: dict, y: np.ndarray, sr: int) -> dict:
    pitch = int(note["pitch"])
    t = float(note["start"])
    decay = _decay(y, sr, t)
    if pitch == 42:
        high = _band(y, sr, t, 6000, 12000)
        mid = _band(y, sr, t, 1500, 5000, dur=0.04)
        lowmid = _band(y, sr, t, 200, 3000, dur=0.2)
        if decay > 0.35 and lowmid > 0.45 * max(high, 1e-12):
            return _retag(note, 49, "crash")
        if decay > 0.18:
            return _retag(note, 46, "open_hat")
        if mid > 3.0 * max(high, 1e-12) and decay < 0.08:
            return _retag(note, 44, "pedal_hat")
        return note
    if pitch == 38:
        body = _band(y, sr, t, 140, 280, dur=0.05)
        click = _band(y, sr, t, 800, 4000, dur=0.03)
        noise = _band(y, sr, t, 2000, 8000, dur=0.05)
        tone = _band(y, sr, t, 80, 400, dur=0.12)
        if tone > 3.0 * max(noise, 1e-12) and decay > 0.12:
            return _retag(note, _tom_pitch(y, sr, t), "tom")
        if click > 4.0 * max(body, 1e-12) and decay < 0.05:
            flat = _flatness(y, sr, t, 800, 5000)
            return _retag(note, 39 if flat > 0.45 else 37, "clap" if flat > 0.45 else "sidestick")
        return note
    if pitch == 36:
        # Same window for both bands so a short sub kick is not beaten by a longer tom window.
        sub = _band(y, sr, t, 30, 85, dur=0.05)
        tom = _band(y, sr, t, 100, 280, dur=0.05)
        f0 = _peak_hz(y, sr, t, 30, 400)
        if 90 <= f0 <= 400 and tom > 2.5 * max(sub, 1e-12) and decay > 0.08:
            return _retag(note, _tom_pitch(y, sr, t), "tom")
    return note


def _retag(note: dict, pitch: int, flag: str) -> dict:
    flags = list(note.get("flags") or [])
    if flag not in flags:
        flags.append(flag)
    note["pitch"] = int(pitch)
    note["flags"] = flags
    return note


def _tom_pitch(y: np.ndarray, sr: int, t: float) -> int:
    f0 = _peak_hz(y, sr, t, 80, 420)
    if f0 < 140:
        return 45
    if f0 < 220:
        return 47
    return 50


def _residual_toms(y: np.ndarray, sr: int, existing: list[dict], params: dict[str, Any]) -> list[dict]:
    """Tonal onsets the three templates did not already claim, in the tom band."""
    import librosa

    hop = HOP
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band = (freqs >= 90) & (freqs < 320)
    if not band.any() or spec.shape[1] < 8:
        return []
    env = spec[band].mean(axis=0)
    nov = np.maximum(0.0, np.diff(env, prepend=env[:1]))
    scale = float(np.percentile(nov, 99)) + 1e-9
    nov = nov / scale
    delta = 0.35 + 0.2 * float(params.get("onset_threshold", 0.5))
    wait = max(4, int(0.08 * sr / hop))
    peaks = librosa.util.peak_pick(nov, pre_max=3, post_max=3, pre_avg=16, post_avg=4, delta=delta, wait=wait)
    taken = [float(n["start"]) for n in existing]
    fps = sr / hop
    out = []
    for p in peaks:
        t = float(p / fps)
        if any(abs(t - s) < 0.05 for s in taken):
            continue
        col = spec[band, p]
        if float(col.max()) < 4.0 * float(np.median(col) + 1e-12):
            continue
        if float(env[p]) < 0.35 * float(np.max(env) + 1e-9):
            continue
        f0 = float(freqs[band][int(np.argmax(col))])
        pitch = 45 if f0 < 140 else 47 if f0 < 220 else 50
        level = float(np.clip(env[p] / (float(np.max(env)) + 1e-9), 0, 1))
        out.append({
            "id": _nid(), "pitch": pitch, "start": round(t, 4), "end": round(t + 0.12, 4),
            "velocity": int(np.clip(round(40 + 80 * level), 1, 127)),
            "confidence": round(float(np.clip(level, 0, 1)), 3), "bends": None,
            "flags": ["residual_tom"],
        })
    return out


def _nid() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


def _slice(y: np.ndarray, sr: int, t: float, dur: float) -> np.ndarray:
    i0 = max(0, int(t * sr))
    i1 = min(len(y), i0 + max(16, int(dur * sr)))
    return y[i0:i1]


def _spectrum(seg: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    if seg.size < 32:
        return np.zeros(2), np.zeros(2)
    window = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    return freqs, spec


def _band(y: np.ndarray, sr: int, t: float, lo: float, hi: float, dur: float = 0.06) -> float:
    freqs, spec = _spectrum(_slice(y, sr, t, dur), sr)
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return 0.0
    return float(spec[m].sum())


def _peak_hz(y: np.ndarray, sr: int, t: float, lo: float, hi: float) -> float:
    freqs, spec = _spectrum(_slice(y, sr, t, 0.12), sr)
    m = (freqs >= lo) & (freqs < hi)
    if not m.any() or spec[m].max() <= 0:
        return (lo + hi) / 2
    return float(freqs[m][int(np.argmax(spec[m]))])


def _flatness(y: np.ndarray, sr: int, t: float, lo: float, hi: float) -> float:
    freqs, spec = _spectrum(_slice(y, sr, t, 0.04), sr)
    m = (freqs >= lo) & (freqs < hi)
    band = spec[m]
    band = band[band > 1e-12]
    if band.size < 4:
        return 0.0
    geo = float(np.exp(np.mean(np.log(band))))
    arith = float(np.mean(band))
    return geo / arith if arith > 0 else 0.0


def _decay(y: np.ndarray, sr: int, t: float, thresh: float = 0.25, max_s: float = 0.6) -> float:
    seg = np.abs(_slice(y, sr, t, max_s))
    if seg.size < 8:
        return 0.0
    win = max(1, int(0.008 * sr))
    smooth = np.convolve(seg, np.ones(win) / win, mode="same")
    peak_i = int(np.argmax(smooth[: max(1, int(0.03 * sr))]))
    peak = float(smooth[peak_i])
    if peak < 1e-5:
        return 0.0
    tail = smooth[peak_i:]
    below = np.where(tail < thresh * peak)[0]
    if below.size == 0:
        return max_s
    return float(below[0] / sr)
