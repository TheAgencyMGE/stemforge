"""Assign notes transcribed from the full mix to the separated source that plays them.

A mix-level transcriber sees every instrument at once, so its notes have no reliable owner. Each
note goes to the source with the most constant-Q energy at that note's pitch during its first
250 ms. When the runner-up source is within ``margin`` of the winner, the note is still assigned
to the winner but flagged ``attribution_uncertain`` so the editor can show it.
"""

from __future__ import annotations

import numpy as np

from .audio import to_mono


def attribute_notes(notes: list[dict], sources: dict[str, np.ndarray], sr: int,
                    margin: float = 1.5) -> dict[str, list[dict]]:
    import librosa

    if not sources:
        return {}
    hop, rate = 256, 22050
    fps = rate / hop
    cqt = {}
    for name, x in sources.items():
        y = librosa.resample(to_mono(x).astype(np.float32), orig_sr=sr, target_sr=rate, res_type="soxr_hq")
        cqt[name] = np.abs(librosa.cqt(y, sr=rate, hop_length=hop, fmin=librosa.midi_to_hz(21), n_bins=88,
                                       bins_per_octave=12))
    out: dict[str, list[dict]] = {name: [] for name in sources}
    for n in notes:
        row = int(np.clip(int(n["pitch"]) - 21, 0, 87))
        i0 = int(float(n["start"]) * fps)
        i1 = max(i0 + 2, min(int(float(n["end"]) * fps), i0 + int(0.25 * fps)))
        energy = {k: float(c[row, i0:i1].mean()) if i0 < c.shape[1] else 0.0 for k, c in cqt.items()}
        ranked = sorted(energy, key=lambda k: -energy[k])
        note = dict(n)
        if len(ranked) > 1 and energy[ranked[0]] < margin * energy[ranked[1]]:
            note["flags"] = sorted({*(note.get("flags") or []), "attribution_uncertain"})
        out[ranked[0]].append(note)
    return out
