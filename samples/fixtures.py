"""Deterministic evaluation fixtures with exact ground truth (CC0-1.0, like the demo).

Each fixture is synthesised on demand from the demo's voices, so no extra audio is committed.
They are synthetic: they test that discovery and transcription behave correctly on known
content (absence, quiet notes, repeated notes, silence, reverb, chunk boundaries). They do not
predict accuracy on commercial recordings.

    fixture(name) -> (mix (2, n) float32 at 44.1 kHz, truth dict)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

import generate_demo as demo

SR = demo.SR


def _reverb(x: np.ndarray, seconds: float = 2.5, wet: float = 0.35, seed: int = 7) -> np.ndarray:
    """Exponentially decaying noise impulse response (RT60 ~ ``seconds``), applied per channel."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir /= np.sqrt(np.sum(ir ** 2))
    out = np.zeros((x.shape[0], x.shape[1] + n - 1))
    for c in range(x.shape[0]):
        out[c] = fftconvolve(x[c], ir)
    dry = np.zeros_like(out)
    dry[:, : x.shape[1]] = x
    return (1 - wet) * dry + wet * out


def keys_solo() -> tuple[np.ndarray, dict]:
    """Solo keys: chords, a melody with repeated notes, very quiet notes, 2.5 s of silence first,
    and a reverb tail at the end. Nothing else plays."""
    beat = demo.BEAT
    start = 2.5
    notes: list[dict] = []
    length = start + 8 * 4 * beat + 4.0
    mix = np.zeros((2, int(length * SR)))
    for bar in range(8):
        t_bar = start + bar * 4 * beat
        _, _, chord = demo.CHORDS[bar % 4]
        for m in chord:  # whole-bar chord, two octaves around middle C
            demo.add(mix, demo.keys_note(m, 4 * beat * 0.95), t_bar, 0.2)
            notes.append({"pitch": m, "start": t_bar, "end": t_bar + 4 * beat * 0.95, "gain": 1.0})
        top = chord[-1] + 12
        for k in range(4):  # repeated quarter notes on one pitch: must stay four notes
            t = t_bar + k * beat
            gain = 0.15 if bar % 4 == 3 else 1.0  # every fourth bar is ~16 dB quieter
            demo.add(mix, gain * demo.keys_note(top, beat * 0.8), t, -0.2)
            notes.append({"pitch": top, "start": t, "end": t + beat * 0.8, "gain": gain})
    mix = _reverb(mix)[:, : int(length * SR)]
    mix /= np.max(np.abs(mix)) / 0.89
    for n in notes:
        n["start"], n["end"] = round(n["start"], 4), round(n["end"], 4)
    truth = {"bpm": demo.BPM, "key": "A minor", "license": "CC0-1.0", "duration": length,
             "notes": {"keys": notes},
             "expect": {"present": ["keyboards"], "absent": ["vocals", "drums"]},
             "silence": {"leading": start, "reverb_tail_after": start + 8 * 4 * beat}}
    return mix.astype(np.float32), truth


def drums_bass() -> tuple[np.ndarray, dict]:
    """Instrumental: drums and bass only, with silence before and after."""
    beat = demo.BEAT
    start, bars = 1.0, 6
    length = start + bars * 4 * beat + 3.0
    mix = np.zeros((2, int(length * SR)))
    truth: dict[str, list] = {"drums": [], "bass": []}
    for bar in range(bars):
        t_bar = start + bar * 4 * beat
        _, root, _ = demo.CHORDS[bar % 4]
        for b8 in range(8):
            t = t_bar + b8 * beat / 2
            if b8 in (0, 4):
                demo.add(mix, demo.kick(), t)
                truth["drums"].append({"pitch": 36, "start": t, "end": t + 0.06})
            if b8 in (2, 6):
                demo.add(mix, demo.snare(), t, 0.1)
                truth["drums"].append({"pitch": 38, "start": t, "end": t + 0.06})
            if b8 not in (2, 6):
                demo.add(mix, demo.hihat(), t, -0.3)
                truth["drums"].append({"pitch": 42, "start": t, "end": t + 0.06})
            m = root + (12 if b8 == 6 else 0)
            demo.add(mix, demo.bass_note(m, beat / 2 * 0.9), t)
            truth["bass"].append({"pitch": m, "start": t, "end": t + beat / 2 * 0.9})
    mix /= np.max(np.abs(mix)) / 0.89
    for part in truth.values():
        for n in part:
            n["start"], n["end"] = round(n["start"], 4), round(n["end"], 4)
    return mix.astype(np.float32), {"bpm": demo.BPM, "license": "CC0-1.0", "duration": length, "notes": truth,
                                    "expect": {"present": ["drums", "bass"], "absent": ["vocals", "keyboards"]}}


def full_demo() -> tuple[np.ndarray, dict]:
    mix, meta = demo.build()
    meta["expect"] = {"present": ["drums", "bass", "keyboards"], "absent": ["vocals"]}
    return mix, meta


def long_demo(repeats: int = 3) -> tuple[np.ndarray, dict]:
    """The demo three times back to back (~95 s): separation runs in several 30 s chunks, so any
    chunk-boundary misalignment shows up as onset errors in the later copies."""
    mix, meta = demo.build()
    n = mix.shape[1]
    out = np.concatenate([mix] * repeats, axis=1)
    notes = {k: [{**x, "start": round(x["start"] + i * n / SR, 4), "end": round(x["end"] + i * n / SR, 4)}
                 for i in range(repeats) for x in v] for k, v in meta["notes"].items()}
    return out, {**meta, "duration": out.shape[1] / SR, "notes": notes,
                 "expect": {"present": ["drums", "bass", "keyboards"], "absent": ["vocals"]}}


FIXTURES = {"demo": full_demo, "keys_solo": keys_solo, "drums_bass": drums_bass, "long_demo": long_demo}


def fixture(name: str) -> tuple[np.ndarray, dict]:
    return FIXTURES[name]()
