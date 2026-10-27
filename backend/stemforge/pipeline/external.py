"""Optional transcription engines that run in their own Python environments.

These engines have licences or dependency stacks that must not enter the main install, so each
one lives in a separate virtualenv and is called as a subprocess. Stemforge only exchanges files
with it (audio in, notes out); no code or weights are imported into this process.

| Engine | What it is | Licence (checked October 2026) | Setup |
|---|---|---|---|
| ``transkun`` | Semi-CRF piano transcription with velocity (Yan & Duan) | Code MIT. The pip package ships its checkpoint with no stated licence. | ``pip install transkun`` in a venv; set ``STEMFORGE_TRANSKUN_PYTHON``. |
| ``yourmt3`` | YourMT3+ multi-instrument AMT, YPTF.MoE+Multi (Chang et al., 2024) | Code GPL-3.0 (run as a separate program, not linked). Checkpoint: Apache-2.0 per its Hugging Face card. Loaded by the authors' code as a full pickle, inside its own environment only. | Checkout of the ``mimbres/YourMT3`` space plus the checkpoint; set ``STEMFORGE_YOURMT3_PYTHON`` and ``STEMFORGE_YOURMT3_DIR``. |
| ``muscriptor`` | Multi-instrument transformer AMT (Kyutai + Mirelo, 2026) | Code MIT. Weights **CC BY-NC 4.0** plus the model card's conditions (research, rights to the input audio), gated on Hugging Face. | ``pip install muscriptor`` in a venv, accept the model conditions on Hugging Face, ``hf auth login``; set ``STEMFORGE_MUSCRIPTOR_PYTHON``. |

Neither engine reports a per-note probability, so notes they produce have ``confidence: null``
with the flag ``no_model_confidence``. A score is never invented for them. Velocity comes from the
engine when it predicts one (Transkun) and otherwise from the stem's own loudness at the onset.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from .audio import to_mono, write_wav

log = logging.getLogger(__name__)


class ExternalEngineError(Exception):
    """The engine is not installed, not licensed on this machine, or failed. Message is shown to users."""


@dataclass(frozen=True)
class ExternalEngine:
    id: str
    label: str
    setting: str  # Settings attribute holding the interpreter path
    licence: str
    timeout_s: int


EXTERNAL: dict[str, ExternalEngine] = {
    "transkun": ExternalEngine(
        "transkun", "Transkun (piano, with velocity)", "transkun_python",
        "Code MIT; the bundled checkpoint has no stated licence. Used here for research.", 3600),
    "yourmt3": ExternalEngine(
        "yourmt3", "YourMT3+ (multi-instrument)", "yourmt3_python",
        "Code GPL-3.0 (separate program); checkpoint Apache-2.0 per its model card. Research use.", 7200),
    "muscriptor": ExternalEngine(
        "muscriptor", "MuScriptor (multi-instrument)", "muscriptor_python",
        "Code MIT; weights CC BY-NC 4.0 with Kyutai/Mirelo conditions (non-commercial research, rights to the "
        "input audio). Gated on Hugging Face.", 7200),
}

# MuScriptor instrument groups -> General MIDI program (drums handled separately). The group list
# is the model's own (`muscriptor list-instruments`); the GM numbers are the usual family members.
MUSCRIPTOR_GM: dict[str, int] = {
    "acoustic_piano": 0, "electric_piano": 4, "chromatic_percussion": 11, "organ": 16, "acoustic_guitar": 25,
    "clean_electric_guitar": 27, "distorted_electric_guitar": 30, "acoustic_bass": 32, "electric_bass": 33,
    "violin": 40, "viola": 41, "cello": 42, "contrabass": 43, "orchestral_harp": 46, "timpani": 47,
    "string_ensemble": 48, "synth_strings": 50, "voice": 52, "orchestra_hit": 55, "trumpet": 56,
    "trombone": 57, "tuba": 58, "french_horn": 60, "brass_section": 61, "soprano_and_alto_sax": 65,
    "tenor_sax": 66, "baritone_sax": 67, "oboe": 68, "english_horn": 69, "bassoon": 70, "clarinet": 71,
    "flutes": 73, "synth_lead": 81, "synth_pad": 89,
}

# Stemforge registry ids -> MuScriptor groups, used to condition the model on what discovery found.
REGISTRY_TO_MUSCRIPTOR: dict[str, list[str]] = {
    "drum_kit": ["drums"], "drum_machine": ["drums"], "bass": ["electric_bass", "acoustic_bass"],
    "electric_bass": ["electric_bass"], "upright_bass": ["acoustic_bass"], "synth_bass": ["electric_bass", "synth_lead"],
    "guitar": ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
    "acoustic_guitar": ["acoustic_guitar"], "electric_guitar": ["clean_electric_guitar", "distorted_electric_guitar"],
    "piano": ["acoustic_piano"], "electric_piano": ["electric_piano"], "organ": ["organ"],
    "synthesizer": ["synth_lead", "synth_pad"], "synth_lead": ["synth_lead"], "synth_pad": ["synth_pad"],
    "mallet": ["chromatic_percussion"], "bells": ["chromatic_percussion"], "violin": ["violin"], "viola": ["viola"],
    "cello": ["cello"], "string_ensemble": ["string_ensemble", "synth_strings"], "double_bass_orch": ["contrabass"],
    "harp": ["orchestral_harp"], "timpani": ["timpani"], "trumpet": ["trumpet"], "trombone": ["trombone"],
    "tuba": ["tuba"], "french_horn": ["french_horn"], "brass_section": ["brass_section"],
    "saxophone": ["soprano_and_alto_sax", "tenor_sax", "baritone_sax"], "flute": ["flutes"], "piccolo": ["flutes"],
    "clarinet": ["clarinet"], "oboe": ["oboe"], "bassoon": ["bassoon"],
    "woodwind_section": ["flutes", "clarinet", "oboe", "bassoon"],
    "lead_vocal": ["voice"], "choir": ["voice"], "orchestra": ["string_ensemble", "brass_section", "flutes", "timpani"],
}


def interpreter(engine_id: str) -> Path | None:
    value = getattr(get_settings(), EXTERNAL[engine_id].setting, None)
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def status(engine_id: str) -> dict[str, Any]:
    eng = EXTERNAL[engine_id]
    py = interpreter(engine_id)
    return {"id": eng.id, "label": eng.label, "licence": eng.licence, "installed": py is not None,
            "detail": None if py else f"Set STEMFORGE_{eng.setting.upper()} to the engine's own Python."}


def _run(engine_id: str, args: list[str]) -> subprocess.CompletedProcess:
    py = interpreter(engine_id)
    if py is None:
        raise ExternalEngineError(status(engine_id)["detail"] or "not installed")
    proc = subprocess.run([str(py), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=EXTERNAL[engine_id].timeout_s, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        low = tail.lower()
        if "gated" in low or ("huggingface" in low and ("401" in tail or "403" in tail)):
            raise ExternalEngineError(
                f"{EXTERNAL[engine_id].label}: the weights are gated. Accept the model conditions on Hugging Face "
                "with your account and run `hf auth login` in the engine's environment.")
        raise ExternalEngineError(f"{EXTERNAL[engine_id].label} failed: {tail.strip().splitlines()[-1] if tail.strip() else proc.returncode}")
    return proc


def _device() -> str:
    """The app's DEVICE setting, in the form both engines' CLIs accept."""
    return {"cuda": "cuda", "cpu": "cpu"}.get(get_settings().device, "auto")


def _velocity_at(y: np.ndarray, sr: int):
    from .transcription import _velocity_curve

    return _velocity_curve(y, sr)


def _nid() -> str:
    return uuid.uuid4().hex[:12]


def transkun_notes(audio: np.ndarray, sr: int) -> list[dict]:
    """Piano notes with Transkun's own velocities. Times are seconds from the start of ``audio``."""
    import mido

    with tempfile.TemporaryDirectory(prefix="sf-transkun-") as td:
        wav, mid = Path(td) / "in.wav", Path(td) / "out.mid"
        write_wav(wav, audio, sr, subtype="PCM_24")
        dev = "cuda" if _device() == "cuda" else "cpu"  # Transkun has no "auto"
        _run("transkun", ["-m", "transkun.transcribe", str(wav), str(mid), "--device", dev])
        notes = _midi_notes(mido.MidiFile(str(mid)))
    for n in notes:
        n["flags"] = ["no_model_confidence"]
    return notes


def muscriptor_notes(audio: np.ndarray, sr: int, instruments: list[str] | None = None,
                     model: str | None = None) -> list[dict]:
    """Multi-instrument notes. Each note carries ``instrument`` (MuScriptor group) and ``program``.

    MuScriptor emits no velocity, so velocity is read from ``audio``'s loudness at each onset.
    ``instruments`` restricts decoding to those groups (the model's own conditioning option).
    """
    settings = get_settings()
    model = model or settings.muscriptor_model
    with tempfile.TemporaryDirectory(prefix="sf-muscriptor-") as td:
        wav, out = Path(td) / "in.wav", Path(td) / "out.json"
        write_wav(wav, audio, sr, subtype="PCM_24")
        args = ["-m", "muscriptor", "transcribe", str(wav), "--format", "json", "--output", str(out),
                "--model", model, "--device", _device(), "--no-quantize"]
        if instruments:
            args += ["--instruments", ",".join(sorted(set(instruments)))]
        _run("muscriptor", args)
        events = json.loads(out.read_text(encoding="utf-8"))
    return notes_from_muscriptor_events(events, audio, sr)


def yourmt3_notes(audio: np.ndarray, sr: int) -> list[dict]:
    """Multi-instrument notes from YourMT3+. Each note carries its GM ``program`` and ``is_drum``."""
    code_dir = get_settings().yourmt3_dir
    if not code_dir or not Path(code_dir).is_dir():
        raise ExternalEngineError("Set STEMFORGE_YOURMT3_DIR to the YourMT3 checkout (code plus checkpoint).")
    runner = Path(__file__).with_name("engine_runners") / "yourmt3_run.py"
    with tempfile.TemporaryDirectory(prefix="sf-yourmt3-") as td:
        wav, out = Path(td) / "in.wav", Path(td) / "out.json"
        write_wav(wav, audio, sr, subtype="PCM_24")
        _run("yourmt3", [str(runner), str(code_dir), str(wav), str(out)])
        doc = json.loads(out.read_text(encoding="utf-8"))
    notes = []
    for n in doc["notes"]:
        start, end = float(n["start"]), max(float(n["end"]), float(n["start"]) + 0.01)
        if n["is_drum"]:
            end = start + 0.06
        notes.append({"id": _nid(), "pitch": int(n["pitch"]), "start": round(start, 4), "end": round(end, 4),
                      "velocity": int(np.clip(n["velocity"], 1, 127)), "confidence": None, "bends": None,
                      "program": None if n["is_drum"] else int(n["program"]), "is_drum": bool(n["is_drum"]),
                      "flags": ["no_model_confidence"]})
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def notes_from_muscriptor_events(events: list[dict], audio: np.ndarray, sr: int) -> list[dict]:
    """Pair MuScriptor's start/end events into notes (exposed for tests)."""
    import librosa

    y = librosa.resample(to_mono(audio).astype(np.float32), orig_sr=sr, target_sr=22050, res_type="soxr_hq")
    vel = _velocity_at(y, 22050)
    starts: dict[int, dict] = {}
    notes: list[dict] = []
    for ev in events:
        if ev.get("type") == "start":
            starts[int(ev["index"])] = ev
        elif ev.get("type") == "end":
            s = starts.pop(int(ev["start_event_index"]), None)
            if s is not None:
                notes.append(_mus_note(s, float(ev["end_time"]), vel))
    for s in starts.values():  # drum hits and notes left open at the end carry no end event
        end = float(s["start_time"]) + (0.06 if s.get("instrument") == "drums" else 0.25)
        notes.append(_mus_note(s, end, vel))
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def _mus_note(start_ev: dict, end: float, vel) -> dict:
    start = float(start_ev["start_time"])
    end = max(end, start + 0.01)
    inst = str(start_ev.get("instrument", ""))
    return {
        "id": _nid(), "pitch": int(start_ev["pitch"]), "start": round(start, 4), "end": round(end, 4),
        "velocity": vel(start, end), "confidence": None, "bends": None, "instrument": inst,
        "program": None if inst == "drums" else MUSCRIPTOR_GM.get(inst), "is_drum": inst == "drums",
        "flags": ["no_model_confidence"],
    }


def _midi_notes(mf) -> list[dict]:
    """Notes from a MIDI file in seconds, with velocity. Iterating a MidiFile yields delta times in
    seconds with every tempo change applied, so tempo maps are honoured."""
    notes: list[dict] = []
    open_: dict[tuple[int, int], tuple[float, int]] = {}
    t = 0.0
    for msg in mf:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_[(msg.channel, msg.note)] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            started = open_.pop((msg.channel, msg.note), None)
            if started is not None:
                notes.append({"id": _nid(), "pitch": int(msg.note), "start": round(started[0], 4),
                              "end": round(max(t, started[0] + 0.01), 4), "velocity": int(started[1]),
                              "confidence": None, "bends": None})
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes
