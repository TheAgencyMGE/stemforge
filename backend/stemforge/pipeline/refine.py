"""Instrument-specific review of transcribed notes.

Corrections that change a pitch keep the previous pitch on the note (``raw_pitch``) and a flag.
Nothing in this module deletes a note. Suggestions are hypotheses for the editor, not facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio import to_mono
from .instruments import STEM_PITCH_RANGE

# Low E, A, D, G, B, high E. String 6 is index 0.
GUITAR_TUNING = (40, 45, 50, 55, 59, 64)
_PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_CHORD_TEMPLATES: dict[tuple[int, ...], str] = {
    (0, 4, 7): "major",
    (0, 3, 7): "minor",
    (0, 4, 7, 10): "dominant 7",
    (0, 3, 7, 10): "minor 7",
    (0, 4, 7, 11): "major 7",
    (0, 3, 6): "diminished",
    (0, 4, 8): "augmented",
    (0, 5, 7): "suspended 4",
    (0, 2, 7): "suspended 2",
}


@dataclass
class Refined:
    notes: list[dict[str, Any]]
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    tablature: dict[str, Any] | None = None


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def apply(stem: str, engine: str, notes: list[dict], audio: np.ndarray | None, sr: int) -> Refined:
    copied = [dict(n) for n in notes]
    suggestions: list[dict[str, Any]] = []
    if stem == "bass" and engine != "drum-nmf" and audio is not None and copied:
        copied, octave_notes = correct_bass_octaves(copied, audio, sr)
        if octave_notes:
            suggestions.append({
                "code": "octave_corrected",
                "severity": "review",
                "message": (
                    f"{_count(len(octave_notes), 'bass note')} "
                    f"{'was' if len(octave_notes) == 1 else 'were'} moved down an octave because the subharmonic "
                    "was stronger than the written pitch. The previous pitch is kept on each note and "
                    "can be restored. This is a correction, not a measurement of the player's intent."
                ),
                "note_ids": [n["id"] for n in octave_notes],
            })
    suggestions.extend(review_notes(stem, copied))
    tab = tablature_estimate(copied) if stem == "guitar" else None
    if stem in {"piano", "guitar"}:
        chord = chord_estimate(copied)
        if chord is not None:
            suggestions.append(chord)
    return Refined(copied, suggestions, tab)


def correct_bass_octaves(notes: list[dict], audio: np.ndarray, sr: int) -> tuple[list[dict], list[dict]]:
    """Move a bass note down an octave when the spectrum's subharmonic is the stronger peak.

    Only notes at MIDI 48 or above are eligible, and the destination must stay inside E1–E4.
    """
    y = to_mono(audio)
    changed: list[dict] = []
    for note in notes:
        pitch = int(note["pitch"])
        if pitch < 48 or pitch - 12 < 28:
            continue
        # Moving onto a note that already occupies the lower pitch at this onset would make
        # the two events the same MIDI note, and the file writer keeps only one of them.
        if any(
            int(other["pitch"]) == pitch - 12 and abs(float(other["start"]) - float(note["start"])) < 0.03
            for other in notes if other is not note
        ):
            continue
        if _subharmonic_wins(y, sr, float(note["start"]), pitch):
            note["raw_pitch"] = pitch
            note["pitch"] = pitch - 12
            flags = list(note.get("flags") or [])
            if "octave_corrected" not in flags:
                flags.append("octave_corrected")
            note["flags"] = flags
            changed.append(note)
    return notes, changed


def _subharmonic_wins(y: np.ndarray, sr: int, t: float, pitch: int) -> bool:
    import librosa

    fundamental = float(librosa.midi_to_hz(pitch))
    below = fundamental / 2.0
    if below < 30:
        return False
    i0 = max(0, int(t * sr))
    n = int(0.2 * sr)
    seg = y[i0:i0 + n]
    if seg.size < n // 2:
        return False
    window = np.hanning(seg.size)
    spec = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)

    def mag(freq: float) -> float:
        k = int(np.argmin(np.abs(freqs - freq)))
        return float(spec[max(0, k - 1): k + 2].max())

    def is_peak(freq: float) -> bool:
        k = int(np.argmin(np.abs(freqs - freq)))
        if k <= 0 or k >= len(spec) - 1:
            return False
        return bool(spec[k] >= spec[k - 1] and spec[k] >= spec[k + 1])

    return bool(is_peak(below) and mag(below) > 1.55 * max(mag(fundamental), 1e-12))


def review_notes(stem: str, notes: list[dict]) -> list[dict[str, Any]]:
    if stem.startswith("kit_"):
        return []
    suggestions: list[dict[str, Any]] = []
    if not notes:
        return suggestions
    span = STEM_PITCH_RANGE.get(stem)
    if span is not None and stem != "drums":
        outside = [n["id"] for n in notes if not span[0] <= int(n["pitch"]) <= span[1]]
        if outside:
            suggestions.append({
                "code": "pitch_range",
                "severity": "review",
                "message": (
                    f"{_count(len(outside), 'note')} "
                    f"{'sits' if len(outside) == 1 else 'sit'} outside the usual MIDI range for this stem "
                    f"({span[0]}–{span[1]}). {'It was' if len(outside) == 1 else 'They were'} kept."
                ),
                "note_ids": outside[:80],
            })
        jumps = _octave_jumps(notes)
        if jumps:
            suggestions.append({
                "code": "octave_jump",
                "severity": "review",
                "message": (
                    f"{_count(len(jumps), 'note')} "
                    f"{'sits' if len(jumps) == 1 else 'sit'} an octave away from the notes around "
                    f"{'it' if len(jumps) == 1 else 'them'} and "
                    f"{'is' if len(jumps) == 1 else 'are'} not doubling another pitch. "
                    f"{'It was' if len(jumps) == 1 else 'They were'} kept. "
                    "A leap and a harmonic error look the same here."
                ),
                "note_ids": jumps[:80],
            })
    duplicates = _duplicates(notes)
    if duplicates:
        suggestions.append({
            "code": "duplicate",
            "severity": "review",
                "message": (
                f"{_count(len(duplicates), 'note')} "
                f"{'starts' if len(duplicates) == 1 else 'start'} within 30 ms of another note at the same pitch. "
                "Both were kept. The MIDI writer shortens a note that would overlap the next attack."
            ),
            "note_ids": duplicates[:80],
        })
    fragments = _fragments(notes)
    if fragments:
        suggestions.append({
            "code": "fragmented",
            "severity": "review",
                "message": (
                f"{_count(len(fragments), 'very short note')} of the same pitch "
                f"{'sits' if len(fragments) == 1 else 'sit'} next to each other. "
                "They may be one sustain that was split, or real re-attacks. They were kept."
            ),
            "note_ids": fragments[:80],
        })
    long_ids = _long_sustains(stem, notes)
    if long_ids:
        suggestions.append({
            "code": "long_sustain",
            "severity": "review",
                "message": (
                f"{_count(len(long_ids), 'note')} "
                f"{'is' if len(long_ids) == 1 else 'are'} unusually long for this stem and not strongly voiced. "
                f"{'It was' if len(long_ids) == 1 else 'They were'} kept."
            ),
            "note_ids": long_ids[:80],
        })
    dense = _dense(stem, notes)
    if dense is not None:
        suggestions.append(dense)
    return suggestions


def _octave_jumps(notes: list[dict]) -> list[str]:
    ordered = sorted(notes, key=lambda n: (n["start"], n["pitch"]))
    flagged: list[str] = []
    for note in ordered:
        near = [n for n in ordered if n is not note and abs(float(n["start"]) - float(note["start"])) <= 0.35]
        if not near:
            continue
        if any(abs(int(n["pitch"]) - int(note["pitch"])) <= 2 and _overlaps(n, note) for n in near):
            continue
        if any(int(n["pitch"]) == int(note["pitch"]) - 12 and _overlaps(n, note) for n in near):
            continue
        median = float(np.median([int(n["pitch"]) for n in near]))
        if int(note["pitch"]) - median >= 12:
            flagged.append(note["id"])
    return flagged


def _overlaps(a: dict, b: dict) -> bool:
    return float(a["start"]) < float(b["end"]) and float(b["start"]) < float(a["end"])


def _duplicates(notes: list[dict]) -> list[str]:
    by_pitch: dict[int, list[dict]] = {}
    for note in notes:
        by_pitch.setdefault(int(note["pitch"]), []).append(note)
    flagged: list[str] = []
    for group in by_pitch.values():
        group.sort(key=lambda n: float(n["start"]))
        for prev, note in zip(group, group[1:], strict=False):
            if abs(float(note["start"]) - float(prev["start"])) <= 0.03 and _overlaps(prev, note):
                flagged.append(note["id"])
    return flagged


def _fragments(notes: list[dict]) -> list[str]:
    by_pitch: dict[int, list[dict]] = {}
    for note in notes:
        if float(note["end"]) - float(note["start"]) <= 0.09:
            by_pitch.setdefault(int(note["pitch"]), []).append(note)
    flagged: list[str] = []
    for group in by_pitch.values():
        group.sort(key=lambda n: float(n["start"]))
        run: list[dict] = []
        for note in group:
            if run and float(note["start"]) - float(run[-1]["end"]) > 0.04:
                if len(run) >= 4:
                    flagged.extend(n["id"] for n in run)
                run = []
            run.append(note)
        if len(run) >= 4:
            flagged.extend(n["id"] for n in run)
    return flagged


def _long_sustains(stem: str, notes: list[dict]) -> list[str]:
    if stem == "drums":
        return []
    limit = {"bass": 4.0, "vocals": 6.0}.get(stem, 8.0)
    out = []
    for note in notes:
        dur = float(note["end"]) - float(note["start"])
        conf = note.get("confidence")
        weak = conf is not None and float(conf) < 0.35
        if dur > limit and (weak or dur > limit + 4):
            out.append(note["id"])
    return out


def _dense(stem: str, notes: list[dict]) -> dict[str, Any] | None:
    if stem == "drums":
        cap = 40
    elif stem in {"vocals", "bass"}:
        cap = 16
    else:
        cap = 40
    buckets: dict[int, int] = {}
    for note in notes:
        buckets[int(float(note["start"]))] = buckets.get(int(float(note["start"])), 0) + 1
    if not buckets:
        return None
    worst = max(buckets.values())
    if worst <= cap:
        return None
    return {
        "code": "dense",
        "severity": "review",
        "message": (
            f"One second of this stem contains {worst} note onsets. That can be a real texture "
            "or a cloud of false notes. Nothing was removed."
        ),
        "note_ids": [],
    }


def chord_estimate(notes: list[dict]) -> dict[str, Any] | None:
    """Name simultaneous pitch-class sets that match a small triad/seventh template."""
    ordered = sorted(notes, key=lambda n: float(n["start"]))
    used: set[str] = set()
    labels: list[str] = []
    for note in ordered:
        if note["id"] in used:
            continue
        group = [n for n in ordered if abs(float(n["start"]) - float(note["start"])) <= 0.04]
        for n in group:
            used.add(n["id"])
        pcs = sorted({int(n["pitch"]) % 12 for n in group})
        if len(pcs) < 3:
            continue
        root = min(group, key=lambda n: int(n["pitch"]))
        root_pc = int(root["pitch"]) % 12
        rel = tuple(sorted((pc - root_pc) % 12 for pc in pcs))
        name = _CHORD_TEMPLATES.get(rel)
        if name:
            labels.append(f"{_PC[root_pc]} {name}")
    if not labels:
        return None
    shown = ", ".join(labels[:8])
    extra = f" (+{len(labels) - 8} more)" if len(labels) > 8 else ""
    return {
        "code": "chord_template",
        "severity": "info",
        "message": (
            f"{len(labels)} groups match a pitch-class template ({shown}{extra}). "
            "This is not a chord-listening model, and the notes were not replaced by chord symbols."
        ),
        "note_ids": [],
    }


def tablature_estimate(notes: list[dict], max_fret: int = 15) -> dict[str, Any]:
    """Assign each note to a standard-tuning string. Collisions are left unassigned.

    The result is a fret estimate. It is not a transcription of the player's fingering,
    bends, or which string was actually used.
    """
    events: list[dict[str, Any]] = []
    groups = _onset_groups(notes, window=0.035)
    unassigned = 0
    for group in groups:
        frets_used: dict[int, int] = {}
        ordered = sorted(group, key=lambda n: int(n["pitch"]))
        for note in ordered:
            choice = _best_string(int(note["pitch"]), frets_used, max_fret)
            if choice is None:
                unassigned += 1
                events.append({
                    "note_id": note["id"], "start": note["start"], "pitch": int(note["pitch"]),
                    "string": None, "fret": None,
                })
                continue
            string_index, fret = choice
            frets_used[string_index] = fret
            events.append({
                "note_id": note["id"], "start": note["start"], "pitch": int(note["pitch"]),
                "string": 6 - string_index, "fret": fret,
            })
    return {
        "estimate": True,
        "tuning": "EADGBE",
        "tuning_midi": list(GUITAR_TUNING),
        "disclaimer": (
            "Standard-tuning fret estimate. String choice prefers a free string and a lower fret. "
            "Bends, slides and the real fingering are not detected."
        ),
        "assigned": len(events) - unassigned,
        "unassigned": unassigned,
        "events": events[:200],
    }


def _best_string(pitch: int, used: dict[int, int], max_fret: int) -> tuple[int, int] | None:
    candidates: list[tuple[int, int, int]] = []
    for index, open_pitch in enumerate(GUITAR_TUNING):
        fret = pitch - open_pitch
        if index in used or not 0 <= fret <= max_fret:
            continue
        # Prefer lower frets, then a lower string (index 0) when frets tie.
        candidates.append((fret, index, fret))
    if not candidates:
        return None
    candidates.sort()
    _, index, fret = candidates[0]
    return index, fret


def _onset_groups(notes: list[dict], window: float) -> list[list[dict]]:
    ordered = sorted(notes, key=lambda n: float(n["start"]))
    groups: list[list[dict]] = []
    for note in ordered:
        if groups and float(note["start"]) - float(groups[-1][0]["start"]) <= window:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def engine_disagreements(primary: list[dict], secondary: list[dict], secondary_engine: str,
                         onset_tol: float = 0.06) -> list[dict[str, Any]]:
    """Notes the primary engine wrote that the comparison engine does not support.

    The primary notes are kept. Disagreement is not treated as an error.
    """
    orphans = []
    for note in primary:
        supported = False
        for other in secondary:
            if abs(float(note["start"]) - float(other["start"])) > onset_tol:
                continue
            if abs(int(note["pitch"]) - int(other["pitch"])) <= 1:
                supported = True
                break
        if not supported:
            orphans.append(note["id"])
    if not orphans:
        return []
    return [{
        "code": "engine_disagreement",
        "severity": "review",
        "message": (
            f"{len(orphans)} notes from the primary engine have no onset within {int(onset_tol * 1000)} ms "
            f"and one semitone in {secondary_engine}. They were kept. The engines are not averaged."
        ),
        "note_ids": orphans[:80],
    }]


def comparison_engine(stem: str, primary: str) -> str | None:
    """A second engine that answers a different question than the primary, or None."""
    if stem in {"drums"} or stem.startswith("kit_"):
        return None
    if stem == "bass":
        return "pyin" if primary == "basic-pitch" else "basic-pitch" if primary == "pyin" else None
    if stem == "vocals":
        return "basic-pitch" if primary == "pyin" else "pyin" if primary == "basic-pitch" else None
    if stem in {"guitar", "piano", "other", "mix"} and primary == "basic-pitch":
        return None  # pYIN on a polyphonic stem mostly deletes chords; it is not a useful comparison
    return None
