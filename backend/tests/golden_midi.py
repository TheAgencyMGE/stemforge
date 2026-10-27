"""The Soundtrap-style MIDI fixture shared by backend and frontend parser tests."""

from __future__ import annotations

from pathlib import Path

from stemforge.pipeline import midi

GOLDEN = Path(__file__).resolve().parents[2] / "frontend" / "tests" / "fixtures" / "soundtrap-golden.mid"
BPM = 96  # an integer tempo, as the Soundtrap package writes it
OFFSET = 2.5  # processed region starts 2.5 s into the original file

KEYS = [
    {"pitch": 60, "start": 0.0, "end": 0.48, "velocity": 100},
    {"pitch": 64, "start": 0.5, "end": 0.98, "velocity": 64},
    {"pitch": 67, "start": 1.0123, "end": 1.9, "velocity": 30, "bends": [[0.0, 0.0], [0.2, 0.5], [0.4, 1.0]]},
]
DRUMS = [{"pitch": 36, "start": 0.0, "end": 0.06, "velocity": 110},
         {"pitch": 38, "start": 0.625, "end": 0.685, "velocity": 90},
         {"pitch": 42, "start": 0.3125, "end": 0.3725, "velocity": 70}]


def specs() -> list[midi.MidiTrackSpec]:
    return [midi.MidiTrackSpec("Acoustic piano", KEYS, program=0),
            midi.MidiTrackSpec("Drum machine", DRUMS, is_drum=True)]


def build() -> bytes:
    import io

    buf = io.BytesIO()
    midi.build_midi(specs(), BPM, "4/4", OFFSET).save(file=buf)
    return buf.getvalue()


if __name__ == "__main__":
    GOLDEN.write_bytes(build())
    print(f"wrote {GOLDEN}")
