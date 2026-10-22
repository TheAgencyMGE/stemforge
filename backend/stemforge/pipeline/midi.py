"""Standard MIDI File writing (mido) and independent validation (symusic, a C++ parser)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido

PPQ = 480
BEND_RANGE = 2  # semitones; set explicitly via RPN 0 on every channel that uses bends
KEY_NAMES = {  # mido key signature names
    ("C", "major"): "C", ("C#", "major"): "C#", ("D", "major"): "D", ("Eb", "major"): "Eb",
    ("E", "major"): "E", ("F", "major"): "F", ("F#", "major"): "F#", ("G", "major"): "G",
    ("Ab", "major"): "Ab", ("A", "major"): "A", ("Bb", "major"): "Bb", ("B", "major"): "B",
    ("C", "minor"): "Cm", ("C#", "minor"): "C#m", ("D", "minor"): "Dm", ("Eb", "minor"): "Ebm",
    ("E", "minor"): "Em", ("F", "minor"): "Fm", ("F#", "minor"): "F#m", ("G", "minor"): "Gm",
    ("Ab", "minor"): "G#m", ("A", "minor"): "Am", ("Bb", "minor"): "Bbm", ("B", "minor"): "Bm",
}


class MidiValidationError(Exception):
    pass


@dataclass
class MidiTrackSpec:
    name: str
    notes: list[dict[str, Any]]
    program: int = 0
    is_drum: bool = False


def _sec_to_tick(t: float, bpm: float) -> int:
    return max(0, int(round(t * bpm / 60.0 * PPQ)))


def _channels(specs: list[MidiTrackSpec]) -> list[int]:
    chans, nxt = [], 0
    for s in specs:
        if s.is_drum:
            chans.append(9)
        else:
            if nxt == 9:
                nxt += 1
            chans.append(nxt % 16 if nxt < 16 else 15)
            nxt += 1
    return chans


def build_midi(
    specs: list[MidiTrackSpec], bpm: float, time_signature: str = "4/4", offset: float = 0.0,
    key: dict | None = None, markers: list[dict] | None = None,
) -> mido.MidiFile:
    """Type-1 SMF: track 0 holds tempo/meter/key/markers; one track per stem.

    `offset` (seconds) is added to every event so MIDI aligns with the *original* file's timeline
    even when only a trimmed section was processed.
    """
    bpm = float(bpm) if bpm and bpm > 0 else 120.0
    mf = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Stemforge", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    num, den = (int(x) for x in time_signature.split("/"))
    meta.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))
    if key and (key.get("tonic"), key.get("mode")) in KEY_NAMES:
        meta.append(mido.MetaMessage("key_signature", key=KEY_NAMES[(key["tonic"], key["mode"])], time=0))
    abs_meta = []
    for m in markers or []:
        abs_meta.append((_sec_to_tick(m["start"] + offset, bpm), mido.MetaMessage("marker", text=str(m["label"]))))
    last = 0
    for tick, msg in sorted(abs_meta, key=lambda x: x[0]):
        meta.append(msg.copy(time=tick - last))
        last = tick
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mf.tracks.append(meta)

    for spec, ch in zip(specs, _channels(specs), strict=True):
        mf.tracks.append(_build_track(spec, ch, bpm, offset))
    return mf


def _build_track(spec: MidiTrackSpec, ch: int, bpm: float, offset: float) -> mido.MidiTrack:
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=spec.name[:64], time=0))
    events: list[tuple[int, int, mido.Message]] = []  # (tick, order, msg); note-offs sort before note-ons
    if not spec.is_drum:
        events.append((0, 0, mido.Message("program_change", channel=ch, program=int(spec.program) & 127)))
    uses_bends = (not spec.is_drum) and any(n.get("bends") for n in spec.notes)
    if uses_bends:
        for ctl, val in ((101, 0), (100, 0), (6, BEND_RANGE), (38, 0), (101, 127), (100, 127)):
            events.append((0, 1, mido.Message("control_change", channel=ch, control=ctl, value=val)))
    for n, on, off in _tick_notes(spec.notes, bpm, offset):
        pitch = int(n["pitch"])
        vel = int(min(127, max(1, round(n.get("velocity", 90)))))
        events.append((on, 3, mido.Message("note_on", channel=ch, note=pitch, velocity=vel)))
        events.append((off, 2, mido.Message("note_off", channel=ch, note=pitch, velocity=0)))
        if uses_bends and n.get("bends"):
            for t_rel, semis in n["bends"]:
                t = _sec_to_tick(float(n["start"]) + offset + float(t_rel), bpm)
                t = min(max(t, on), off)
                val = int(round(max(-BEND_RANGE, min(BEND_RANGE, float(semis))) / BEND_RANGE * 8191))
                events.append((t, 4 if t == on else 1, mido.Message("pitchwheel", channel=ch, pitch=max(-8192, min(8191, val)))))
            events.append((off, 1, mido.Message("pitchwheel", channel=ch, pitch=0)))
    events.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tick, _, msg in events:
        tr.append(msg.copy(time=tick - last))
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _tick_notes(notes: list[dict[str, Any]], bpm: float, offset: float) -> list[tuple[dict, int, int]]:
    """Convert to ticks and make same-pitch notes unambiguous: drop duplicates starting on the same
    tick and end each note no later than the next onset of that pitch (overlapping note-ons on one
    channel are paired differently by different parsers)."""
    by_pitch: dict[int, list[tuple[dict, int, int]]] = {}
    for n in notes:
        pitch = int(n["pitch"])
        if not 0 <= pitch <= 127:
            continue
        on = _sec_to_tick(float(n["start"]) + offset, bpm)
        off = max(on + 1, _sec_to_tick(float(n["end"]) + offset, bpm))
        by_pitch.setdefault(pitch, []).append((n, on, off))
    out = []
    for items in by_pitch.values():
        items.sort(key=lambda x: x[1])
        for i, (n, on, off) in enumerate(items):
            if i > 0 and items[i - 1][1] == on:
                continue
            nxt = next((it[1] for it in items[i + 1:] if it[1] > on), None)
            if nxt is not None:
                off = min(off, nxt)
            out.append((n, on, off))
    out.sort(key=lambda x: x[1])
    return out


def normalized_note_count(notes: list[dict[str, Any]], bpm: float, offset: float = 0.0) -> int:
    return len(_tick_notes(notes, bpm, offset))


def save_and_validate(mf: mido.MidiFile, path: Path, specs: list[MidiTrackSpec], bpm: float,
                      offset: float = 0.0) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mf.save(str(path))
    return validate(path, specs, bpm, offset)


def validate(path: Path, specs: list[MidiTrackSpec], bpm: float, offset: float = 0.0) -> dict[str, Any]:
    """Re-open with symusic (independent of mido) and check track names, note counts, pitches and onsets."""
    from symusic import Score

    bpm = float(bpm) if bpm and bpm > 0 else 120.0
    try:
        score = Score(str(path), ttype="tick")
    except Exception as exc:  # pragma: no cover - would indicate a writer bug
        raise MidiValidationError(f"independent parser rejected file: {exc}") from exc
    by_name = {t.name: t for t in score.tracks}
    report: dict[str, Any] = {"parser": "symusic", "ticks_per_quarter": score.ticks_per_quarter, "tracks": []}
    if score.ticks_per_quarter != PPQ:
        raise MidiValidationError("ticks_per_quarter mismatch")
    for s in specs:
        expected = _tick_notes(s.notes, bpm, offset)
        t = by_name.get(s.name[:64])
        got = len(t.notes) if t is not None else 0
        if t is None and expected:
            raise MidiValidationError(f"track {s.name!r} missing after re-parse")
        if got != len(expected):
            raise MidiValidationError(f"track {s.name!r}: wrote {len(expected)} notes, parser read {got}")
        if t is not None and expected:
            want = sorted((on, int(n["pitch"])) for n, on, _ in expected)
            have = sorted((int(n.time), int(n.pitch)) for n in t.notes)
            if want != have:
                raise MidiValidationError(f"track {s.name!r}: onset/pitch mismatch after re-parse")
        report["tracks"].append({"name": s.name, "notes": got, "is_drum": bool(t.is_drum) if t is not None else s.is_drum})
    return report
