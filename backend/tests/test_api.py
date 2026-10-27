"""API integration tests (FastAPI TestClient against a real SQLite DB and local storage)."""

from __future__ import annotations

import io
import json
import zipfile

import pretty_midi
import pytest

from tests.conftest import SAMPLES, requires_models, write_tone


def upload(client, path, name=None, **params):
    return client.post(
        "/api/projects", params=params, content=path.read_bytes(),
        headers={"X-Filename": name or path.name, "Content-Type": "application/octet-stream"},
    )


def test_health_and_system(client):
    assert client.get("/api/healthz").json()["status"] == "ok"
    sysinfo = client.get("/api/system").json()
    assert {e["id"] for e in sysinfo["engines"]} == {"basic-pitch", "pyin", "drum-nmf", "onset-events"}
    assert "wav" in sysinfo["formats"]
    assert client.get("/api/openapi.json").status_code == 200


def test_rejects_non_audio(client, tmp_path):
    exe = tmp_path / "song.mp3"
    exe.write_bytes(b"MZ\x90\x00" + b"\x00" * 200)  # an executable renamed to .mp3
    r = upload(client, exe)
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_format"


def test_rejects_empty_and_too_large(client, tmp_path, monkeypatch):
    empty = tmp_path / "e.wav"
    empty.write_bytes(b"")
    assert upload(client, empty).status_code == 400
    from stemforge import config

    monkeypatch.setenv("STEMFORGE_MAX_UPLOAD_MB", "1")
    config.get_settings.cache_clear()
    big = write_tone(tmp_path / "big.wav", seconds=8)  # ~2.8 MB
    r = upload(client, big)
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_rejects_corrupt_wav_and_bad_trim(client, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 40)
    assert upload(client, bad).status_code == 422
    tone = write_tone(tmp_path / "t.wav", seconds=3)
    r = upload(client, tone, trim_start=5)
    assert r.status_code == 422 and r.json()["error"]["code"] == "bad_trim"


def test_upload_sanitises_name_and_enqueues(client, tmp_path):
    tone = write_tone(tmp_path / "t.wav", seconds=2)
    r = upload(client, tone, name="..%2F..%2Fevil%20name.wav")
    assert r.status_code == 201, r.text
    p = r.json()
    assert p["original_filename"] == "evil name.wav"
    assert p["status"] == "queued" and p["job"]["status"] == "queued"
    # The source is private: not listed among downloadable artifacts.
    assert all(a["kind"] != "source" for a in p["artifacts"])
    assert client.get(f"/api/projects/{p['id']}").status_code == 200
    assert client.delete(f"/api/projects/{p['id']}").status_code == 204
    assert client.get(f"/api/projects/{p['id']}").status_code == 404


def test_signed_url_rejects_tampering(client):
    r = client.get("/api/files/abc", params={"exp": 9999999999, "sig": "0" * 64})
    assert r.status_code == 403


def test_api_token_enforced(env, monkeypatch):
    from fastapi.testclient import TestClient

    from stemforge import config
    from stemforge.api.app import create_app

    monkeypatch.setenv("STEMFORGE_API_TOKEN", "s3cret")
    config.get_settings.cache_clear()
    with TestClient(create_app()) as c:
        assert c.get("/api/projects").status_code == 401
        assert c.get("/api/projects", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/healthz").status_code == 200


def test_bundle_import_rejects_zip_slip(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.txt", "x")
        z.writestr("project.json", "{}")
    r = client.post("/api/projects/import", content=buf.getvalue())
    assert r.status_code == 422 and "unexpected entry" in r.json()["error"]["message"]


def test_instrument_catalog_and_upload_options(client, tmp_path):
    catalog = client.get("/api/instruments")
    assert catalog.status_code == 200
    ids = {row["id"] for family in catalog.json()["families"] for row in family["instruments"]}
    assert "violin" in ids and "lead_vocal" in ids
    missing = client.get("/api/instruments/not_an_instrument")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "unknown_instrument"
    violin = client.get("/api/instruments/violin").json()
    assert violin["separate"] == "unavailable" and violin["model"] is None
    tone = write_tone(tmp_path / "opt.wav", seconds=2)
    r = upload(client, tone, drum_kit="true", transcribe_mix="true", compare_engines="true")
    assert r.status_code == 201, r.text
    opts = r.json()["options"]
    assert opts["drum_kit"] is True and opts["transcribe_mix"] is True and opts["compare_engines"] is True


# ---------------------------------------------------------------- full pipeline with real models


def _run_worker_until_idle():
    from stemforge.worker import process_one

    n = 0
    while process_one("test-worker"):
        n += 1
    return n


@requires_models
@pytest.mark.models
@pytest.mark.slow
def test_full_pipeline_demo_song(client, tmp_path):
    demo = SAMPLES / "stemforge-demo.flac"
    r = upload(client, demo)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    assert _run_worker_until_idle() == 1
    p = client.get(f"/api/projects/{pid}").json()
    assert p["status"] == "ready", p["error"]
    assert p["job"]["status"] == "succeeded"
    assert set(p["job"]["stages"]) == {"decode", "analyze", "separate", "render", "transcribe", "midi"}
    tracks = {t["stem"]: t for t in p["tracks"]}
    # The synthetic demo has no voice: no vocals track may be created at all, not even a silent one.
    # With the tagger installed the keys are also extracted as piano; without it they stay in "other".
    assert {"drums", "bass"} <= set(tracks) <= {"drums", "bass", "piano", "other"}
    assert "vocals" not in tracks
    assert tracks["drums"]["engine"] == "drum-nmf" and tracks["drums"]["note_count"] > 50
    assert tracks["bass"]["note_count"] > 40
    checks = [e for e in p["analysis"]["discovery"]["log"] if e["step"] == "family_check"]
    assert next(e for e in checks if e["stem"] == "vocals")["accepted"] is False
    for t in tracks.values():
        assert t["stats"]["source"]["activity"]["active_seconds"] > 0.75
    kinds = {(a["kind"], a["stem"]) for a in p["artifacts"]}
    for s in tracks:
        assert ("stem", s) in kinds and ("preview", s) in kinds and ("peaks", s) in kinds
    assert ("stem", "vocals") not in kinds
    assert ("midi_combined", None) in kinds

    # Signed download of a stem works and is the full-length 44.1 kHz audio.
    stem = next(a for a in p["artifacts"] if a["kind"] == "stem" and a["stem"] == "bass")
    dl = client.get(stem["url"])
    assert dl.status_code == 200
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(dl.content))
    assert sr == 44100 and abs(len(data) / sr - p["analysis"]["duration"]) < 0.01

    # Edit notes with optimistic concurrency, then export reflects the edit.
    bass = tracks["bass"]
    doc = client.get(f"/api/projects/{pid}/tracks/{bass['id']}/notes").json()
    new_notes = doc["notes"][:10] + [{"id": "user1", "pitch": 100, "start": 1.0, "end": 1.5, "velocity": 64,
                                      "confidence": None, "bends": None}]
    ok = client.put(f"/api/projects/{pid}/tracks/{bass['id']}/notes", json={"base_version": doc["version"], "notes": new_notes})
    assert ok.status_code == 200
    stale = client.put(f"/api/projects/{pid}/tracks/{bass['id']}/notes", json={"base_version": doc["version"], "notes": []})
    assert stale.status_code == 409
    bad = client.put(f"/api/projects/{pid}/tracks/{bass['id']}/notes",
                     json={"base_version": ok.json()["version"], "notes": [{"id": "x", "pitch": 60, "start": 2, "end": 1, "velocity": 64}]})
    assert bad.status_code == 422

    mid = client.get(f"/api/projects/{pid}/export/midi", params={"stems": "bass"})
    assert mid.status_code == 200
    pm = pretty_midi.PrettyMIDI(io.BytesIO(mid.content))
    assert len(pm.instruments) == 1 and len(pm.instruments[0].notes) == 11
    assert any(n.pitch == 100 for n in pm.instruments[0].notes)

    multi = pretty_midi.PrettyMIDI(io.BytesIO(client.get(f"/api/projects/{pid}/export/midi").content))
    assert {i.name for i in multi.instruments} == {t["display_name"] for t in tracks.values()}
    assert next(i for i in multi.instruments if i.name == tracks["drums"]["display_name"]).is_drum

    # Re-transcribe one track with a different engine (cached features path for the second run).
    rt = client.post(f"/api/projects/{pid}/tracks/{bass['id']}/transcribe", json={"engine": "pyin", "params": {"min_freq": 30, "max_freq": 500}})
    assert rt.status_code == 202
    assert _run_worker_until_idle() == 1
    assert client.get(f"/api/jobs/{rt.json()['id']}").json()["status"] == "succeeded"
    t2 = next(t for t in client.get(f"/api/projects/{pid}").json()["tracks"] if t["stem"] == "bass")
    assert t2["engine"] == "pyin" and t2["note_count"] > 20 and t2["engine_reason"] == "Selected manually"

    # Full project bundle round-trip restores tracks, edits and audio artifacts.
    bundle = client.get(f"/api/projects/{pid}/export/bundle", params={"full": "true"})
    assert bundle.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(bundle.content)).namelist()
    assert "project.json" in names and "multitrack.mid" in names and any(n.startswith("stems/") for n in names)
    restored = client.post("/api/projects/import", content=bundle.content)
    assert restored.status_code == 201, restored.text
    rp = restored.json()
    assert rp["status"] == "ready" and len(rp["tracks"]) == len(tracks)
    rbass = next(t for t in rp["tracks"] if t["stem"] == "bass")
    assert rbass["note_count"] == t2["note_count"]
    assert any(a["kind"] == "preview" for a in rp["artifacts"])

    # Resuming a finished project re-uses every stage.
    rep = client.post(f"/api/projects/{pid}/reprocess", json={})
    assert rep.status_code == 202
    _run_worker_until_idle()
    job = client.get(f"/api/jobs/{rep.json()['id']}").json()
    assert job["status"] == "succeeded"
    assert job["stages"]["decode"]["status"] == "skipped" and job["stages"]["separate"]["status"] == "skipped"


@requires_models
@pytest.mark.models
def test_trimmed_section_offsets_midi(client, tmp_path):
    demo = SAMPLES / "stemforge-demo.flac"
    r = upload(client, demo, trim_start=10, trim_end=16)
    assert r.status_code == 201
    pid = r.json()["id"]
    _run_worker_until_idle()
    p = client.get(f"/api/projects/{pid}").json()
    assert p["status"] == "ready" and p["offset_seconds"] == 10
    assert abs(p["analysis"]["duration"] - 6.0) < 0.01
    pm = pretty_midi.PrettyMIDI(io.BytesIO(client.get(f"/api/projects/{pid}/export/midi", params={"stems": "drums"}).content))
    starts = [n.start for n in pm.instruments[0].notes]
    assert starts and min(starts) >= 10.0 - 1e-3 and max(starts) <= 16.0 + 1e-3
    truth = json.loads((SAMPLES / "stemforge-demo.truth.json").read_text())["notes"]["drums"]
    kicks = [n["start"] for n in truth if n["pitch"] == 36 and 10.05 < n["start"] < 15.9]
    got = [n.start for n in pm.instruments[0].notes if n.pitch == 36]
    assert sum(any(abs(g - k) < 0.05 for g in got) for k in kicks) >= len(kicks) - 1
