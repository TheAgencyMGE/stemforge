"""External research engines (through stub engines on the real subprocess path) and the Soundtrap package."""

from __future__ import annotations

import io
import json
import sys
import textwrap
import zipfile

import mido
import numpy as np
import pytest
import soundfile as sf

from stemforge import config
from stemforge.pipeline import external, transcription

SR = 44100

MUSCRIPTOR_STUB = '''
import json, sys
argv = sys.argv[1:]
out = argv[argv.index("--output") + 1]
if __import__("os").environ.get("STUB_GATED"):
    sys.stderr.write("401 Client Error: this repo is gated. You must accept the conditions\\n"); sys.exit(1)
events = [
    {"type": "start", "pitch": 60, "start_time": 0.5, "index": 0, "instrument": "acoustic_piano"},
    {"type": "start", "pitch": 36, "start_time": 1.0, "index": 1, "instrument": "drums"},
    {"type": "end", "end_time": 1.2, "start_event_index": 0},
    {"type": "start", "pitch": 64, "start_time": 2.0, "index": 2, "instrument": "acoustic_piano"},
    {"type": "end", "end_time": 2.5, "start_event_index": 2},
]
json.dump(events, open(out, "w"))
open(out + ".argv", "w").write(json.dumps(argv))
'''

TRANSKUN_STUB = '''
import sys, mido
wav, out = sys.argv[1], sys.argv[2]
mf = mido.MidiFile(ticks_per_beat=480)
t = mido.MidiTrack(); mf.tracks.append(t)
t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))           # 120 BPM for one beat
t.append(mido.Message("note_on", note=60, velocity=101, time=0))
t.append(mido.MetaMessage("set_tempo", tempo=1000000, time=480))        # then 60 BPM
t.append(mido.Message("note_off", note=60, velocity=0, time=0))
t.append(mido.Message("note_on", note=67, velocity=33, time=480))       # one 60-BPM beat later = 1.5 s
t.append(mido.Message("note_off", note=67, velocity=0, time=240))
mf.save(out)
'''


@pytest.fixture()
def stub_engines(tmp_path, monkeypatch):
    pkg = tmp_path / "stubs"
    (pkg / "muscriptor").mkdir(parents=True)
    (pkg / "muscriptor" / "__init__.py").write_text("")
    (pkg / "muscriptor" / "__main__.py").write_text(textwrap.dedent(MUSCRIPTOR_STUB))
    (pkg / "transkun").mkdir()
    (pkg / "transkun" / "__init__.py").write_text("")
    (pkg / "transkun" / "transcribe.py").write_text(textwrap.dedent(TRANSKUN_STUB))
    monkeypatch.setenv("PYTHONPATH", str(pkg))
    monkeypatch.setenv("STEMFORGE_MUSCRIPTOR_PYTHON", sys.executable)
    monkeypatch.setenv("STEMFORGE_TRANSKUN_PYTHON", sys.executable)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def _loud_then_quiet() -> np.ndarray:
    t = np.arange(3 * SR) / SR
    x = np.sin(2 * np.pi * 262 * t) * np.where(t < 1.5, 0.5, 0.02)
    return np.stack([x, x]).astype(np.float32)


def test_muscriptor_adapter_pairs_events_conditions_and_derives_velocity(stub_engines, monkeypatch):
    captured = {}
    real_run = external._run

    def spy(engine_id, args):
        captured["args"] = args
        return real_run(engine_id, args)

    monkeypatch.setattr(external, "_run", spy)
    notes = external.muscriptor_notes(_loud_then_quiet(), SR, ["acoustic_piano", "drums"])
    args = captured["args"]
    assert args[args.index("--instruments") + 1] == "acoustic_piano,drums"
    assert "--no-quantize" in args and args[args.index("--format") + 1] == "json"
    by_pitch = {n["pitch"]: n for n in notes}
    assert by_pitch[60]["start"] == 0.5 and by_pitch[60]["end"] == 1.2 and by_pitch[60]["program"] == 0
    assert by_pitch[36]["is_drum"] and by_pitch[36]["program"] is None  # onset-only drum hit
    assert all(n["confidence"] is None and "no_model_confidence" in n["flags"] for n in notes)
    assert by_pitch[60]["velocity"] > by_pitch[64]["velocity"]  # loud passage vs quiet passage


def test_gated_weights_give_an_actionable_error(stub_engines, monkeypatch):
    monkeypatch.setenv("STUB_GATED", "1")
    with pytest.raises(external.ExternalEngineError, match="gated"):
        external.muscriptor_notes(_loud_then_quiet(), SR)


def test_transkun_adapter_honours_tempo_changes_and_velocity(stub_engines):
    notes = external.transkun_notes(_loud_then_quiet(), SR)
    a, b = notes
    assert (a["pitch"], a["start"], a["end"], a["velocity"]) == (60, 0.0, 0.5, 101)
    assert (b["pitch"], b["start"], b["end"], b["velocity"]) == (67, 1.5, 2.0, 33)


def test_engines_appear_only_when_installed(stub_engines, monkeypatch):
    assert "transkun" in transcription.allowed_engines("piano", False)
    assert transcription.allowed_engines("percussion", True) == ["onset-events"]
    monkeypatch.setenv("STEMFORGE_TRANSKUN_PYTHON", "")
    config.get_settings.cache_clear()
    assert "transkun" not in transcription.allowed_engines("piano", False)


def test_research_engines_enter_the_policy_only_when_opted_in(stub_engines, monkeypatch):
    assert transcription.choose_engine("piano")[0] == "basic-pitch"
    monkeypatch.setenv("STEMFORGE_RESEARCH_ENGINES", "true")
    config.get_settings.cache_clear()
    eng, _, reason = transcription.choose_engine("piano")
    assert eng == "transkun" and "0.97" in reason
    assert transcription.choose_engine("bass")[0] == "basic-pitch"  # no benchmark supports a change there

    # With YourMT3+ installed too, pitched sources use its mix transcription; bass stays on Basic Pitch,
    # which scored higher there on BabySlakh.
    real = transcription.engine_available
    monkeypatch.setattr(transcription, "engine_available", lambda e: True if e == "yourmt3" else real(e))
    for stem in ("piano", "guitar", "other"):
        eng, params, _ = transcription.choose_engine(stem)
        assert (eng, params["input"]) == ("yourmt3", "mix")
    assert transcription.choose_engine("bass")[0] == "basic-pitch"
    assert transcription.choose_engine("vocals")[0] == "pyin"  # no vocal benchmark: unchanged


def test_external_engine_output_round_trips_through_the_feature_cache(stub_engines):
    params = transcription.resolve_params("transkun", {})
    feats = transcription.compute_features("transkun", _loud_then_quiet(), SR, params)
    buf = io.BytesIO()
    np.savez_compressed(buf, **feats)
    buf.seek(0)
    with np.load(buf, allow_pickle=False) as z:  # the cache never needs pickles
        notes = transcription.notes_from_features("transkun", {k: z[k] for k in z.files}, _loud_then_quiet(), SR, params)
    assert [(n["pitch"], n["velocity"]) for n in notes] == [(60, 101), (67, 33)]
    assert transcription.summarize(notes)["mean_confidence"] is None


# ----------------------------------------------------------------------------- API


def test_percussion_track_can_be_retranscribed(client, env):
    """Regression: drum-flagged tracks used to accept only drum-nmf, so onset-events was rejected."""
    from tests.test_discovery_export import _ready_project, tone

    pid = _ready_project(env, {"percussion": tone(1.0, 3000)}, {})
    from stemforge.db import session_scope
    from stemforge.models import Track

    with session_scope() as db:
        t = db.query(Track).filter_by(project_id=pid).one()
        t.is_drum = True
        db.commit()
        tid = t.id
    track = client.get(f"/api/projects/{pid}").json()["tracks"][0]
    assert track["allowed_engines"] == ["onset-events"]
    ok = client.post(f"/api/projects/{pid}/tracks/{tid}/transcribe", json={"engine": "onset-events"})
    assert ok.status_code == 202, ok.text
    bad = client.post(f"/api/projects/{pid}/tracks/{tid}/transcribe", json={"engine": "basic-pitch"})
    assert bad.status_code in (409, 422)


def test_soundtrap_package_has_wav_stems_integer_tempo_midi_and_instructions(client, env):
    from stemforge.db import session_scope
    from stemforge.models import Project, Track
    from stemforge.pipeline import audio
    from tests.test_discovery_export import _ready_project, tone

    keys = tone(2.0, 220, start=1.0, total=4.0)
    pid = _ready_project(env, {"keys": keys}, {}, offset=0.0)
    notes = [{"id": "a", "pitch": 57, "start": 1.0123, "end": 1.5, "velocity": 90, "confidence": 0.8, "bends": None}]
    with session_scope() as db:
        p = db.get(Project, pid)
        p.tempo_bpm = 96.37
        t = db.query(Track).filter_by(project_id=pid).one()
        t.notes, t.generated_notes, t.transcription_status = notes, notes, "done"
        db.commit()
    r = client.get(f"/api/projects/{pid}/export/soundtrap")
    assert r.status_code == 200, r.text
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert "SOUNDTRAP.md" in names and "01 Keys.wav" in names and "01 Keys.mid" in names
    assert not any(n.endswith(".flac") for n in names)  # Soundtrap does not import FLAC
    man = json.loads(z.read("export_manifest.json"))
    assert man["bpm"] == 96 and "96 BPM" in z.read("SOUNDTRAP.md").decode()
    pcm, sr = sf.read(io.BytesIO(z.read("01 Keys.wav")), dtype="int32", always_2d=True)
    assert sr == SR and np.array_equal(pcm.T >> 8, audio.quantize(keys))  # lossless, full timeline
    mf = mido.MidiFile(file=io.BytesIO(z.read("01 Keys.mid")))
    tempos = [m.tempo for tr in mf.tracks for m in tr if m.type == "set_tempo"]
    assert tempos == [mido.bpm2tempo(96)]
    t = 0.0
    for msg in mf:
        t += msg.time
        if msg.type == "note_on" and msg.velocity:
            assert abs(t - 1.0123) <= 60 / 96 / mf.ticks_per_beat  # exact to one tick at the written tempo


# ----------------------------------------------------------------------------- render fidelity


def _renderer_configured() -> bool:
    import os
    from pathlib import Path

    fs, sf2 = os.environ.get("STEMFORGE_FLUIDSYNTH_PATH", ""), os.environ.get("STEMFORGE_SOUNDFONT_PATH", "")
    return bool(fs and sf2 and Path(fs).is_file() and Path(sf2).is_file())


@pytest.mark.skipif(not _renderer_configured(), reason="FluidSynth and a soundfont are not configured")
def test_render_metrics_separate_pitch_timing_and_instrument_errors(tmp_path):
    from stemforge.pipeline import midi, render

    config.get_settings.cache_clear()

    def mk(notes, name, program=0):
        path = tmp_path / f"{name}.mid"
        midi.build_midi([midi.MidiTrackSpec(name, notes, program)], 120.0).save(str(path))
        return render.render_midi(path, SR, 8.0)

    base = [{"pitch": p, "start": 0.5 * i, "end": 0.5 * i + 0.4, "velocity": 90}
            for i, p in enumerate([60, 64, 67, 72, 67, 64, 60, 62, 65, 69, 72, 69])]
    ref = mk(base, "ref")
    same = render.compare(ref, mk(base, "same"), SR)
    transposed = render.compare(ref, mk([{**n, "pitch": n["pitch"] + 3} for n in base], "up"), SR)
    late = render.compare(ref, mk([{**n, "start": n["start"] + 0.12, "end": n["end"] + 0.12} for n in base], "late"), SR)
    violin = render.compare(ref, mk(base, "violin", program=40), SR)
    assert same["chroma_similarity"] > 0.99 and same["onset_correlation"] > 0.99
    assert transposed["chroma_similarity"] < 0.5 < transposed["onset_correlation"]  # wrong pitches, right rhythm
    assert late["onset_correlation"] < 0.5 < late["chroma_similarity"]  # right pitches, wrong timing
    assert violin["chroma_similarity"] > 0.8 and violin["logmel_l1_db"] > transposed["logmel_l1_db"]  # timbre only


def test_mix_notes_are_attributed_to_the_source_that_plays_them():
    from stemforge.pipeline.attribution import attribute_notes

    t = np.arange(2 * SR) / SR

    def tone(midi_pitch, level):
        x = level * np.sin(2 * np.pi * 440 * 2 ** ((midi_pitch - 69) / 12) * t)
        return np.stack([x, x]).astype(np.float32)

    sources = {"bass": tone(40, 0.4), "keys": tone(64, 0.3) + tone(52, 0.2)}
    sources["bass"] = sources["bass"] + tone(52, 0.19)  # pitch 52 is in both, nearly as loud
    notes = [{"id": "a", "pitch": 40, "start": 0.2, "end": 1.0}, {"id": "b", "pitch": 64, "start": 0.2, "end": 1.0},
             {"id": "c", "pitch": 52, "start": 0.2, "end": 1.0}]
    out = attribute_notes(notes, sources, SR)
    assert [n["id"] for n in out["bass"]] == ["a"]
    assert sorted(n["id"] for n in out["keys"]) == ["b", "c"]
    assert "attribution_uncertain" in next(n for n in out["keys"] if n["id"] == "c")["flags"]
    assert "flags" not in next(n for n in out["keys"] if n["id"] == "b")
