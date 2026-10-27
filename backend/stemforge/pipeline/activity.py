"""Frame-level activity: when a source is audible, and where its leading/trailing silence ends.

Two separate questions are answered here and they use different references:

* ``source_activity`` asks whether a stem carries a part of its own. A frame counts as active
  only if the stem is above an absolute floor *and* not negligible next to the mixture at that
  moment, so low-level bleed under a loud mix is not mistaken for a performance while a quiet
  instrument in a quiet passage still counts.
* ``silence_bounds`` asks where an exported file may be trimmed. It looks at the file alone,
  with a low threshold and a release hold so reverb tails and room tone are kept. Only leading
  and trailing silence are ever reported; internal silence is part of the timing and stays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

HOP_SECONDS = 0.01
WIN_SECONDS = 0.05


def frame_db(audio: np.ndarray, sr: int, hop_s: float = HOP_SECONDS, win_s: float = WIN_SECONDS) -> np.ndarray:
    """RMS level per frame in dBFS (channels pooled). Frame i is centred at i * hop."""
    x = audio if audio.ndim == 2 else audio[None, :]
    power = np.mean(x.astype(np.float64) ** 2, axis=0)
    hop = max(1, int(round(hop_s * sr)))
    win = max(hop, int(round(win_s * sr)))
    frames = int(np.ceil(power.size / hop)) if power.size else 0
    if frames == 0:
        return np.zeros(0)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    centers = np.arange(frames) * hop
    lo = np.clip(centers - win // 2, 0, power.size)
    hi = np.clip(centers + win // 2 + 1, 0, power.size)
    mean = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
    return 10 * np.log10(np.maximum(mean, 1e-12))


def frame_peak_db(audio: np.ndarray, sr: int, hop_s: float = HOP_SECONDS) -> np.ndarray:
    x = np.abs(audio if audio.ndim == 2 else audio[None, :]).max(axis=0)
    hop = max(1, int(round(hop_s * sr)))
    frames = int(np.ceil(x.size / hop)) if x.size else 0
    padded = np.zeros(frames * hop, dtype=np.float64)
    padded[: x.size] = x
    return 20 * np.log10(np.maximum(padded.reshape(frames, hop).max(axis=1), 1e-12))


@dataclass
class Activity:
    active_seconds: float
    active_fraction: float
    first_active: float | None
    last_active: float | None
    median_active_db: float | None  # stem level in its active frames (dBFS)
    median_active_rel_db: float | None  # stem minus mixture level in those frames (dB)
    segments: list[tuple[float, float]]  # merged active regions, seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segments"] = [[round(a, 3), round(b, 3)] for a, b in self.segments[:200]]
        return d


def source_activity(stem: np.ndarray, mix: np.ndarray, sr: int, abs_floor_db: float = -60.0,
                    rel_floor_db: float = -40.0, merge_gap_s: float = 0.3) -> Activity:
    """Where ``stem`` is audible as a part of ``mix`` (see module docstring)."""
    s_db = frame_db(stem, sr)
    m_db = frame_db(mix, sr)
    n = min(s_db.size, m_db.size)
    s_db, m_db = s_db[:n], m_db[:n]
    active = (s_db > abs_floor_db) & (s_db - m_db > rel_floor_db)
    total = n * HOP_SECONDS
    if not active.any():
        return Activity(0.0, 0.0, None, None, None, None, [])
    idx = np.flatnonzero(active)
    segments: list[tuple[float, float]] = []
    start = prev = int(idx[0])
    gap = max(1, int(merge_gap_s / HOP_SECONDS))
    for i in idx[1:]:
        if i - prev > gap:
            segments.append((start * HOP_SECONDS, (prev + 1) * HOP_SECONDS))
            start = int(i)
        prev = int(i)
    segments.append((start * HOP_SECONDS, (prev + 1) * HOP_SECONDS))
    return Activity(
        active_seconds=round(float(active.sum()) * HOP_SECONDS, 3),
        active_fraction=round(float(active.mean()), 4),
        first_active=round(float(idx[0]) * HOP_SECONDS, 3),
        last_active=round(float(idx[-1] + 1) * HOP_SECONDS, 3),
        median_active_db=round(float(np.median(s_db[active])), 2),
        median_active_rel_db=round(float(np.median(s_db[active] - m_db[active])), 2),
        segments=segments if total > 0 else [],
    )


@dataclass
class SilenceBounds:
    start_sample: int  # first sample kept
    end_sample: int  # one past the last sample kept
    total_samples: int
    leading_seconds: float
    trailing_seconds: float
    empty: bool

    def to_dict(self) -> dict:
        return asdict(self)


def silence_bounds(audio: np.ndarray, sr: int, threshold_db: float = -60.0, min_silence_s: float = 0.25,
                   pre_roll_s: float = 0.02, post_roll_s: float = 0.25) -> SilenceBounds:
    """Leading/trailing silence of a single file, for trimmed standalone exports.

    A frame is "sound" when its peak exceeds ``threshold_db``. -60 dBFS keeps typical reverb tails
    and room tone; ``post_roll_s`` keeps the release below the threshold, and ``pre_roll_s`` keeps
    a soft attack. Silence shorter than ``min_silence_s`` is not trimmed at all.
    """
    n = audio.shape[-1]
    peaks = frame_peak_db(audio, sr)
    loud = np.flatnonzero(peaks > threshold_db)
    if loud.size == 0:
        return SilenceBounds(0, 0, n, n / sr, 0.0, True)
    hop = int(round(HOP_SECONDS * sr))
    start = max(0, int(loud[0]) * hop - int(pre_roll_s * sr))
    end = min(n, (int(loud[-1]) + 1) * hop + int(post_roll_s * sr))
    if start < int(min_silence_s * sr):
        start = 0
    if n - end < int(min_silence_s * sr):
        end = n
    return SilenceBounds(start, end, n, round(start / sr, 6), round((n - end) / sr, 6), False)
