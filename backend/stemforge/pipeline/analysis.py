"""Musical analysis of the mix: tempo, beat grid, key and coarse section boundaries."""

from __future__ import annotations

import numpy as np

from .audio import level_stats, to_mono

SR = 22050
HOP = 512

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def analyze(audio: np.ndarray, sr: int) -> dict:
    import librosa

    y = librosa.resample(to_mono(audio), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    duration = len(y) / SR
    result: dict = {"duration": round(audio.shape[1] / sr, 4), "sample_rate": sr, **level_stats(audio)}
    if float(np.max(np.abs(y))) < 1e-4:
        result.update({"tempo": None, "beats": [], "key": None, "sections": [], "silent": True})
        return result

    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    # Confidence of the tempo estimate: autocorrelation peak prominence of the onset envelope at the period.
    tempo_conf = _tempo_confidence(onset_env, tempo)
    if len(beats) >= 8:
        # Refine tempo with a least-squares fit of beat time vs. beat index. Individual beat times
        # are quantised to the 23 ms hop; the regression slope averages that error out (a median
        # inter-beat interval would not). Robust to the odd skipped beat via a residual filter.
        idx = np.arange(len(beats))
        slope, icpt = np.polyfit(idx, beats, 1)
        resid = np.abs(beats - (slope * idx + icpt))
        keep = resid < max(0.06, 3 * np.median(resid))
        if keep.sum() >= 8:
            slope, icpt = np.polyfit(idx[keep], beats[keep], 1)
        if slope > 0:
            tempo = 60.0 / float(slope)
    result["tempo"] = round(tempo, 2) if tempo > 0 else None
    result["tempo_confidence"] = round(tempo_conf, 3)
    result["beats"] = [round(float(b), 4) for b in beats]
    result["first_beat"] = round(float(beats[0]), 4) if len(beats) else 0.0

    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    result["key"] = estimate_key(chroma.mean(axis=1))
    result["sections"] = estimate_sections(y, chroma, beat_frames, duration)
    result["silent"] = False
    return result


def _tempo_confidence(onset_env: np.ndarray, tempo: float) -> float:
    if tempo <= 0 or len(onset_env) < 16:
        return 0.0
    env = onset_env - onset_env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lag = int(round(60.0 / tempo * SR / HOP))
    if lag <= 0 or lag >= len(ac):
        return 0.0
    window = ac[max(1, lag - 2): lag + 3]
    return float(np.clip(window.max(), 0.0, 1.0))


def estimate_key(chroma_mean: np.ndarray) -> dict:
    scores = []
    for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(chroma_mean, np.roll(profile, tonic))[0, 1])
            scores.append((r, tonic, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return {
        "tonic": PITCH_CLASSES[best[1]],
        "tonic_pc": best[1],
        "mode": best[2],
        "label": f"{PITCH_CLASSES[best[1]]} {best[2]}",
        "correlation": round(best[0], 3),
        # Margin to the runner-up: small margins mean relative major/minor ambiguity.
        "margin": round(best[0] - second[0], 3),
        "runner_up": f"{PITCH_CLASSES[second[1]]} {second[2]}",
    }


def estimate_sections(y: np.ndarray, chroma: np.ndarray, beat_frames: np.ndarray, duration: float) -> list[dict]:
    """Unsupervised segmentation: beat-synchronous chroma+MFCC, agglomerative clustering in time.

    Labels are letters (A, B, ...) by similarity, not semantic names like "chorus".
    """
    import librosa

    if duration < 20 or len(beat_frames) < 16:
        return [{"start": 0.0, "end": round(duration, 3), "label": "A"}]
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
    n = min(chroma.shape[1], mfcc.shape[1])
    feats = np.vstack([librosa.util.normalize(chroma[:, :n], axis=0), librosa.util.normalize(mfcc[:, :n], axis=0)])
    sync = librosa.util.sync(feats, beat_frames[beat_frames < n], aggregate=np.median)
    k = int(np.clip(round(duration / 30), 2, 10))
    k = min(k, sync.shape[1] - 1)
    if k < 2:
        return [{"start": 0.0, "end": round(duration, 3), "label": "A"}]
    bounds_beats = librosa.segment.agglomerative(sync, k)
    beat_times = librosa.frames_to_time(beat_frames[beat_frames < n], sr=SR, hop_length=HOP)
    bound_times = [0.0] + [float(beat_times[b - 1]) if b > 0 else 0.0 for b in bounds_beats[1:]] + [duration]
    bound_times = sorted(set(round(t, 3) for t in bound_times))
    # Label segments by clustering their mean feature vectors.
    seg_feats = []
    for i in range(len(bound_times) - 1):
        lo = np.searchsorted(beat_times, bound_times[i])
        hi = max(lo + 1, np.searchsorted(beat_times, bound_times[i + 1]))
        seg_feats.append(sync[:, lo:hi].mean(axis=1) if hi <= sync.shape[1] else sync[:, -1])
    labels: list[str] = []
    protos: list[np.ndarray] = []
    for f in seg_feats:
        sims = [float(np.dot(f, p) / (np.linalg.norm(f) * np.linalg.norm(p) + 1e-9)) for p in protos]
        if sims and max(sims) > 0.97:
            labels.append(chr(ord("A") + int(np.argmax(sims))))
        else:
            protos.append(f)
            labels.append(chr(ord("A") + len(protos) - 1))
    return [
        {"start": bound_times[i], "end": bound_times[i + 1], "label": labels[i]}
        for i in range(len(bound_times) - 1)
        if bound_times[i + 1] - bound_times[i] > 0.5
    ]
