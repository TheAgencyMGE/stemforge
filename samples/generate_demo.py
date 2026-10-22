"""Generate the bundled demo song (CC0) together with its ground-truth notes.

Everything here is synthesised from scratch by this script, so the audio is free of third-party
rights and is dedicated to the public domain (CC0 1.0). Because the score is known exactly, the
demo doubles as an evaluation fixture for separation and transcription.

    python samples/generate_demo.py            # writes stemforge-demo.flac + stemforge-demo.truth.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 96.0
BEAT = 60.0 / BPM
BARS = 12
LEN = BARS * 4 * BEAT + 1.5
rng = np.random.default_rng(7)

CHORDS = [("Am", 45, [57, 60, 64]), ("F", 41, [53, 57, 60]), ("C", 48, [55, 60, 64]), ("G", 43, [55, 59, 62])]
MELODY = [  # (beat offset within 4-bar phrase, length in beats, midi)
    (0, 1, 69), (1, 1, 72), (2, 2, 76), (4, 1.5, 77), (5.5, 0.5, 76), (6, 2, 72),
    (8, 1, 72), (9, 1, 74), (10, 2, 76), (12, 1, 74), (13, 1, 71), (14, 2, 67),
]


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def env_adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    t = np.arange(n) / SR
    e = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na, endpoint=False) if na else e[:na]
    e[na:na + nd] = np.linspace(1, s, max(nd, 0), endpoint=False)[: max(0, min(nd, n - na))]
    if nr and n > nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e * (t >= 0)


def add(buf: np.ndarray, x: np.ndarray, t: float, pan: float = 0.0) -> None:
    i = int(round(t * SR))
    j = min(buf.shape[1], i + len(x))
    if j <= i:
        return
    gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[0, i:j] += x[: j - i] * gl
    buf[1, i:j] += x[: j - i] * gr


def onepole_lp(x: np.ndarray, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, lfilter

    b, a = butter(2, cutoff / (SR / 2))
    return lfilter(b, a, x)


def band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(4, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], btype="band", output="sos")
    return sosfilt(sos, x)


def kick() -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 0.9


def snare() -> np.ndarray:
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    noise = band(rng.standard_normal(n), 1200, 6000) * np.exp(-t * 22)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    return 0.55 * noise + 0.4 * body


def hihat() -> np.ndarray:
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    from scipy.signal import butter, sosfilt

    sos = butter(4, 7500 / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, rng.standard_normal(n)) * np.exp(-t * 70) * 0.35


def bass_note(m: int, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = midi_hz(m)
    saw = 2 * ((t * f) % 1.0) - 1
    return onepole_lp(saw, 900) * env_adsr(n, 0.005, 0.1, 0.7, 0.04) * 0.45


def keys_note(m: int, dur: float) -> np.ndarray:
    n = int((dur + 0.3) * SR)
    t = np.arange(n) / SR
    f = midi_hz(m)
    x = sum((0.6 ** k) * np.sin(2 * np.pi * f * (k + 1) * t * (1 + 0.0004 * k)) for k in range(6))
    e = np.exp(-t * 2.2) * env_adsr(n, 0.003, 0.0, 1.0, 0.3)
    e[int(dur * SR):] *= np.exp(-np.arange(n - int(dur * SR)) / SR * 18)
    return x * e * 0.12


def lead_note(m: int, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 0.25 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.3, 0, 1)
    f = midi_hz(m) * 2 ** (vib / 12)
    ph = 2 * np.pi * np.cumsum(f) / SR
    # Vowel-like spectrum: emphasise harmonics near 700 Hz and 1200 Hz formants.
    x = np.zeros(n)
    for k in range(1, 12):
        fk = midi_hz(m) * k
        g = np.exp(-((fk - 700) / 300) ** 2) + 0.6 * np.exp(-((fk - 1200) / 400) ** 2) + 0.15 / k
        x += g * np.sin(k * ph)
    return x * env_adsr(n, 0.04, 0.1, 0.85, 0.08) * 0.16


def build() -> tuple[np.ndarray, dict]:
    n = int(LEN * SR)
    mix = np.zeros((2, n))
    truth: dict[str, list] = {"drums": [], "bass": [], "keys": [], "lead": []}
    start = 0.5  # half a second of lead-in silence
    for bar in range(BARS):
        t_bar = start + bar * 4 * BEAT
        name, root, chord = CHORDS[bar % 4]
        for b8 in range(8):  # drums
            t = t_bar + b8 * BEAT / 2
            if b8 in (0, 4) or (b8 == 5 and bar % 2):
                add(mix, kick(), t)
                truth["drums"].append({"pitch": 36, "start": t, "end": t + 0.06})
            if b8 in (2, 6):
                add(mix, snare(), t, 0.1)
                truth["drums"].append({"pitch": 38, "start": t, "end": t + 0.06})
            if bar >= 1 and b8 not in (2, 6):
                add(mix, hihat(), t, -0.3)
                truth["drums"].append({"pitch": 42, "start": t, "end": t + 0.06})
        if bar >= 1:  # bass: eighth-note pulse on the root, octave jump on beat 4
            for b8 in range(8):
                m = root + (12 if b8 == 6 else 0)
                t = t_bar + b8 * BEAT / 2
                add(mix, bass_note(m, BEAT / 2 * 0.9), t)
                truth["bass"].append({"pitch": m, "start": t, "end": t + BEAT / 2 * 0.9})
        for half in range(2):  # keys: half-note chords
            t = t_bar + half * 2 * BEAT
            for m in chord:
                add(mix, keys_note(m, 2 * BEAT * 0.95), t, 0.25)
                truth["keys"].append({"pitch": m, "start": t, "end": t + 2 * BEAT * 0.95})
    for phrase in (2, 6):  # lead melody enters at bar 3 and bar 7
        t0 = start + phrase * 4 * BEAT
        for off, length, m in MELODY:
            t = t0 + off * BEAT
            add(mix, lead_note(m, length * BEAT * 0.92), t, -0.05)
            truth["lead"].append({"pitch": m, "start": t, "end": t + length * BEAT * 0.92})
    mix /= np.max(np.abs(mix)) / 0.89
    for part in truth.values():
        for note in part:
            note["start"], note["end"] = round(note["start"], 4), round(note["end"], 4)
    meta = {"bpm": BPM, "key": "A minor", "time_signature": "4/4", "duration": LEN, "license": "CC0-1.0",
            "parts": {"drums": "drums", "bass": "bass", "keys": "other", "lead": "vocals"}, "notes": truth}
    return mix.astype(np.float32), meta


def main() -> None:
    out = Path(__file__).resolve().parent
    mix, meta = build()
    sf.write(str(out / "stemforge-demo.flac"), mix.T, SR, subtype="PCM_24")
    (out / "stemforge-demo.truth.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out / 'stemforge-demo.flac'} ({mix.shape[1] / SR:.1f}s)")


if __name__ == "__main__":
    main()
