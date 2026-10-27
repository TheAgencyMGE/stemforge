"""Source discovery, absence detection, silence handling and export behaviour.

The tagger is replaced by a deterministic stub here so the decision logic is tested on its own;
the real PANNs model is exercised by the slow pipeline tests.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from stemforge.pipeline import activity, audio, discovery, separation, tagger
from tests.conftest import SAMPLES, requires_models

SR = 44100


def tone(seconds: float, freq: float, level: float = 0.3, start: float = 0.0, total: float | None = None) -> np.ndarray:
    total = total or start + seconds
    out = np.zeros((2, int(total * SR)), dtype=np.float32)
    t = np.arange(int(seconds * SR)) / SR
    x = (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    i = int(start * SR)
    out[:, i:i + x.size] = x
    return out


# ----------------------------------------------------------------------------- separation hygiene


def test_gate_zeroes_stems_only_where_the_mixture_is_digitally_silent():
    mix = tone(1.0, 220, start=1.0, total=3.0)
    noise = (np.random.default_rng(0).standard_normal(mix.shape) * 1e-3).astype(np.float32)
    stems = {"a": (mix * 0.5 + noise).astype(np.float32), "b": (mix * 0.5 - noise).astype(np.float32)}
    gated = separation.gate_mixture_silence(mix, stems, SR)
    for name in stems:
        assert gated[name].shape == stems[name].shape
        assert np.all(gated[name][:, : int(0.99 * SR)] == 0)  # silent intro: artifacts removed
        assert np.all(gated[name][:, int(2.01 * SR):] == 0)  # silent outro
        mid = slice(int(1.0 * SR) + 10, int(2.0 * SR) - 10)
        assert np.array_equal(gated[name][:, mid], stems[name][:, mid])  # audible samples untouched


def test_gate_leaves_short_gaps_alone():
    mix = tone(2.0, 220)
    mix[:, SR: SR + int(0.02 * SR)] = 0  # 20 ms dropout inside the music: part of the timing
    stems = {"a": mix.copy()}
    assert np.array_equal(separation.gate_mixture_silence(mix, stems, SR)["a"], stems["a"])


# ----------------------------------------------------------------------------- activity and trimming


def test_activity_counts_a_quiet_part_in_a_quiet_passage_but_not_bleed_under_a_loud_one():
    loud = tone(4.0, 110, level=0.5)
    quiet_part = tone(1.0, 880, level=0.01, start=5.0, total=6.0)  # -40 dBFS, alone
    mix = np.concatenate([loud, np.zeros((2, 2 * SR), dtype=np.float32)], axis=1)[:, : quiet_part.shape[1]]
    mix = mix + quiet_part
    bleed = 0.003 * np.concatenate([loud, np.zeros((2, 2 * SR), dtype=np.float32)], axis=1)[:, : mix.shape[1]]
    act_quiet = activity.source_activity(quiet_part, mix, SR)
    act_bleed = activity.source_activity(bleed.astype(np.float32), mix, SR)
    assert act_quiet.active_seconds > 0.9 and abs(act_quiet.first_active - 5.0) < 0.05
    assert act_bleed.active_seconds == 0.0


def test_silence_bounds_keep_reverb_tail_and_internal_silence():
    sr = SR
    x = np.zeros((2, 6 * sr), dtype=np.float32)
    note = tone(0.5, 440, level=0.5)[:, : int(0.5 * sr)]
    x[:, sr: sr + note.shape[1]] = note  # starts at 1.0 s
    x[:, 3 * sr: 3 * sr + note.shape[1]] = note  # second note after 1.5 s of silence
    t = np.arange(int(1.5 * sr)) / sr
    tail = (0.5 * np.exp(-t * 5.5) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # ~ -60 dB at 1.25 s
    x[:, int(3.5 * sr): int(3.5 * sr) + tail.size] += tail
    b = activity.silence_bounds(x, sr)
    assert not b.empty
    assert 0.95 < b.start_sample / sr < 1.0  # pre-roll before the first attack
    tail_end = 3.5 + np.log(10 ** (60 / 20) * 0.5 / 1.0) / 5.5
    assert b.end_sample / sr > tail_end  # the decay above -60 dBFS is not cut
    assert b.end_sample < x.shape[1]
    assert activity.silence_bounds(np.zeros((2, sr), dtype=np.float32), sr).empty


def test_flac_writer_is_bit_exact_and_counts_clipping(tmp_path):
    x = (np.random.default_rng(1).standard_normal((2, SR)) * 0.1).astype(np.float32)
    x[1, 5] = 1.2
    info = audio.write_flac_verified(tmp_path / "x.flac", x, SR, "PCM_24")
    assert info["clipped_samples"] == 1 and info["bit_depth"] == 24 and info["frames"] == SR
    back, _ = sf.read(tmp_path / "x.flac", dtype="int32", always_2d=True)
    assert np.array_equal(back.T >> 8, audio.quantize(x, "PCM_24"))


# ----------------------------------------------------------------------------- extractors


def test_mask_splits_reconstruct_their_parent_exactly():
    parent = tone(2.0, 220) + tone(2.0, 3000, level=0.1)
    split = discovery.ratio_mask_split(parent, {"low": tone(2.0, 220), "high": tone(2.0, 3000, level=0.1)}, SR)
    assert np.max(np.abs(sum(split.values()) - parent)) < 1e-6
    hp = discovery.hpss_split(parent, SR)
    assert np.max(np.abs(hp["percussion"] + hp["rest"] - parent)) < 1e-6


# ----------------------------------------------------------------------------- discovery decisions


class StubTagger:
    """Tags by frequency content: 220 Hz reads as piano, 880 Hz as singing, noise as drums."""

    def __init__(self, monkeypatch):
        monkeypatch.setattr(tagger, "available", lambda: True)
        monkeypatch.setattr(tagger, "tag_windows", self.tag)

    @staticmethod
    def tag(x, sr, hop_seconds=5.0, batch=8):
        mono = audio.to_mono(x)
        spec = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(mono.size, 1 / sr)
        total = spec.sum() + 1e-12

        def share(lo, hi):
            return float(spec[(freqs >= lo) & (freqs < hi)].sum() / total)

        probs = np.zeros((1, 527), dtype=np.float32)
        probs[0, 137] = 0.8  # Music
        probs[0, 153] = 0.9 * share(200, 240)  # Piano
        probs[0, 27] = 0.9 * share(850, 910)  # Singing
        flat = share(4000, 20000)
        probs[0, 162] = 0.9 * flat  # Drum kit
        return {"probs": probs, "starts": np.array([0.0])}


def _no_extra_separator(model, audio_in, cb):
    raise RuntimeError("not available in this test")


def test_piano_solo_produces_one_named_source_and_nothing_invented(monkeypatch):
    StubTagger(monkeypatch)
    mix = tone(4.0, 220)
    zero = np.zeros_like(mix)
    base = {"vocals": zero, "drums": zero, "bass": zero, "other": mix.copy()}
    d = discovery.discover(mix, SR, base, "htdemucs", _no_extra_separator, lambda f, m: None)
    assert set(d.sources) == {"other"}
    assert d.sources["other"].meta["display"] == "Acoustic piano"
    assert d.sources["other"].status == "accepted"
    checks = {e["stem"]: e["accepted"] for e in d.log if e["step"] == "family_check"}
    assert checks == {"vocals": False, "drums": False, "bass": False}


def test_instrumental_bleed_in_the_vocal_stem_is_reassigned_not_shown(monkeypatch):
    StubTagger(monkeypatch)
    piano = tone(4.0, 220)
    bleed = (0.05 * piano).astype(np.float32)  # what a separator leaks into "vocals"
    mix = piano + bleed
    zero = np.zeros_like(mix)
    base = {"vocals": bleed, "drums": zero, "bass": zero, "other": piano}
    d = discovery.discover(mix, SR, base, "htdemucs", _no_extra_separator, lambda f, m: None)
    assert "vocals" not in d.sources
    assert any(e["step"] == "reassign" and e["from"] == "vocals" and e["to"] == "other" for e in d.log)
    # Nothing was lost: the shown sources still add up to the mixture.
    assert d.reconstruction["unexplained_by_tracks_db"] < -60


def test_real_vocals_are_kept(monkeypatch):
    StubTagger(monkeypatch)
    voice, keys = tone(4.0, 880), tone(4.0, 220)
    mix = voice + keys
    zero = np.zeros_like(mix)
    d = discovery.discover(mix, SR, {"vocals": voice, "drums": zero, "bass": zero, "other": keys},
                           "htdemucs", _no_extra_separator, lambda f, m: None)
    assert {"vocals", "other"} <= set(d.sources)
    assert d.sources["vocals"].meta["family"] == "vocals"


def test_ambiguous_residual_is_unresolved_not_invented(monkeypatch):
    StubTagger(monkeypatch)
    odd = tone(4.0, 1500)  # no stub class responds to 1.5 kHz
    zero = np.zeros_like(odd)
    d = discovery.discover(odd, SR, {"vocals": zero, "drums": zero, "bass": zero, "other": odd},
                           "htdemucs", _no_extra_separator, lambda f, m: None)
    assert d.sources["other"].status == "unresolved"
    assert d.sources["other"].meta["display"] == "Unresolved source"


def test_without_the_tagger_quiet_vocal_bleed_is_still_not_a_track(monkeypatch):
    monkeypatch.setattr(tagger, "available", lambda: False)
    keys = tone(4.0, 220)
    bleed = (0.02 * keys).astype(np.float32)
    zero = np.zeros_like(keys)
    d = discovery.discover(keys + bleed, SR, {"vocals": bleed, "drums": zero, "bass": zero, "other": keys},
                           "htdemucs", _no_extra_separator, lambda f, m: None)
    assert "vocals" not in d.sources
    assert d.tagger == "unavailable"
    assert d.sources["other"].status == "accepted"  # no tags: unidentified, not "ambiguous"
    assert any("energy only" in lim for lim in d.sources["other"].meta["limitations"])


def test_no_fixed_stem_ceiling_in_track_ordering():
    from stemforge.pipeline.runner import _ordered

    stems = {f"src{i:02d}": None for i in range(20)} | {"vocals": None, "drums": None}
    order = _ordered(stems)
    assert len(order) == 22 and order[:2] == ["vocals", "drums"]


# ----------------------------------------------------------------------------- export (API level)


def _ready_project(env, stems: dict[str, np.ndarray], meta: dict[str, dict], offset: float = 0.0):
    """A finished project with the given stem audio, built directly in the DB and storage."""
    from stemforge.db import session_scope
    from stemforge.models import Artifact, Project, Track
    from stemforge.storage import get_storage, sha256_file

    storage = get_storage()
    with session_scope() as db:
        p = Project(name="Export test", original_filename="x.wav", source_format="wav", source_size=1,
                    status="ready", options={}, offset_seconds=offset, analysis={"tempo": 120.0})
        db.add(p)
        db.flush()
        for i, (stem, data) in enumerate(stems.items()):
            local = env / f"{stem}.flac"
            audio.write_flac_verified(local, data)
            key = f"projects/{p.id}/stems/{stem}.flac"
            size = storage.put_file(key, local, "audio/flac")
            db.add(Artifact(project_id=p.id, kind="stem", stem=stem, storage_key=key, filename=f"x-{stem}.flac",
                            content_type="audio/flac", size=size, sha256=sha256_file(local), params_hash="t", meta={}))
            db.add(Track(project_id=p.id, stem=stem, display_name=stem.title(), order_index=i, source_model="t",
                         is_drum=False, gm_program=0, stats=meta.get(stem, {}), warnings=[], notes=[],
                         generated_notes=[], transcription_status="skipped"))
        db.commit()
        return p.id


def _zip(client, pid, **params):
    r = client.get(f"/api/projects/{pid}/export/bundle", params=params)
    assert r.status_code == 200, r.text
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return z, json.loads(z.read("export_manifest.json"))


def test_export_excludes_silent_tracks_by_default_and_trims_exactly(client, env):
    keys = tone(1.0, 220, start=2.0, total=5.0)
    quiet = tone(2.0, 880, level=0.003, start=1.0, total=5.0)  # -50 dBFS: quiet but real
    stems = {"keys": keys, "quiet": quiet, "empty": np.zeros_like(keys), "mystery": tone(1.0, 1500, total=5.0)}
    meta = {"empty": {"silent": True}, "mystery": {"source": {"status": "unresolved"}}}
    pid = _ready_project(env, stems, meta, offset=10.0)

    z, man = _zip(client, pid)
    names = {e["stem"] for e in man["files"]}
    assert names == {"keys", "quiet", "mystery"}  # default: every audible source; silent is opt-in
    assert next(e for e in man["files"] if e["stem"] == "mystery")["status"] == "unresolved"
    for e in man["files"]:
        assert e["frames"] == keys.shape[1] and e["timeline_start_seconds"] == 10.0  # synced: full length

    z, man = _zip(client, pid, mode="trimmed", stems="keys,quiet,empty")
    files = {e["stem"]: e for e in man["files"]}
    assert set(files) == {"keys", "quiet"}
    assert [e["stem"] for e in man["excluded"]] == ["empty"]
    k = files["keys"]
    assert 11.9 < k["timeline_start_seconds"] < 12.0  # original-file time = offset + trim start
    data, _ = sf.read(io.BytesIO(z.read(f"stems/{k['file']}")), dtype="int32", always_2d=True)
    original = audio.quantize(keys)
    start = int(round(k["trimmed_leading_seconds"] * SR))
    assert np.array_equal(data.T >> 8, original[:, start:start + data.shape[0]])  # bit-exact slice
    assert files["quiet"]["frames"] > 1.9 * SR  # a quiet instrument is not trimmed away


def test_rendered_mix_leaves_out_muted_tracks_but_their_stems_stay_available(client, env):
    a, b = tone(2.0, 220), tone(2.0, 880)
    pid = _ready_project(env, {"a": a, "b": b}, {})
    z, man = _zip(client, pid, stems="a,b", mix="a")  # lane "b" muted in the mixer
    assert {e["stem"] for e in man["files"]} == {"a", "b"}
    assert man["mix"]["tracks"] == ["A"] and man["mix"]["normalized"] is False
    mixed, _ = sf.read(io.BytesIO(z.read(f"mix/{man['mix']['file']}")), dtype="int32", always_2d=True)
    assert np.array_equal(mixed.T >> 8, audio.quantize(a))


def test_single_track_download_is_trimmed_with_its_timeline_offset(client, env):
    pid = _ready_project(env, {"keys": tone(1.0, 220, start=2.0, total=4.0)}, {}, offset=1.5)
    track = client.get(f"/api/projects/{pid}").json()["tracks"][0]
    r = client.get(f"/api/projects/{pid}/tracks/{track['id']}/audio")
    assert r.status_code == 200
    assert 3.45 < float(r.headers["x-stemforge-timeline-start"]) < 3.5
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == SR and data.shape[0] < 1.5 * SR
    full = client.get(f"/api/projects/{pid}/tracks/{track['id']}/audio", params={"mode": "synced"})
    assert sf.info(io.BytesIO(full.content)).frames == 4 * SR


# ----------------------------------------------------------------------------- real models


@requires_models
@pytest.mark.models
@pytest.mark.slow
def test_padded_instrumental_has_digital_silence_and_no_vocal_track(client, tmp_path):
    """Regression for the silent/muted-download report, on the real models."""
    demo, sr = sf.read(SAMPLES / "stemforge-demo.flac", always_2d=True)
    padded = np.concatenate([np.zeros((3 * sr, 2)), demo, np.zeros((4 * sr, 2))])
    src = tmp_path / "padded.wav"
    sf.write(src, padded, sr, subtype="PCM_24")
    from stemforge.worker import process_one

    r = client.post("/api/projects", content=src.read_bytes(),
                    headers={"X-Filename": "padded.wav", "Content-Type": "application/octet-stream"})
    pid = r.json()["id"]
    while process_one("t"):
        pass
    p = client.get(f"/api/projects/{pid}").json()
    assert p["status"] == "ready", p["error"]
    assert "vocals" not in {t["stem"] for t in p["tracks"]}
    z, man = _zip(client, pid)
    for e in man["files"]:
        data, _ = sf.read(io.BytesIO(z.read(f"stems/{e['file']}")), always_2d=True)
        assert data.shape[0] == padded.shape[0]  # synced: same timeline as the upload
        assert np.all(data[: int(2.9 * sr)] == 0)  # the silent intro is silent, not separator hiss
    z, man = _zip(client, pid, mode="trimmed")
    for e in man["files"]:
        assert e["timeline_start_seconds"] > 2.9 and e["trimmed_trailing_seconds"] > 3.5


def test_weak_speech_tags_alone_do_not_make_a_vocal_track(monkeypatch):
    """Regression: a "Choir Aahs" pad in the vocal stem scored speech 0.26, singing 0.03."""
    monkeypatch.setattr(tagger, "available", lambda: True)

    def tags(x, sr, hop_seconds=5.0, batch=8):
        probs = np.zeros((1, 527), dtype=np.float32)
        probs[0, 137] = 0.8
        if audio.to_mono(x).std() > 0.05:  # the "vocal" stem: speech-like but not singing
            probs[0, 0], probs[0, 27] = 0.26, 0.03
        probs[0, 153] = 0.6  # piano everywhere else
        return {"probs": probs, "starts": np.array([0.0])}

    monkeypatch.setattr(tagger, "tag_windows", tags)
    pad, keys = tone(4.0, 660, level=0.1), tone(4.0, 220, level=0.02)
    zero = np.zeros_like(pad)
    d = discovery.discover(pad + keys, SR, {"vocals": pad, "drums": zero, "bass": zero, "other": keys},
                           "htdemucs", _no_extra_separator, lambda f, m: None)
    assert "vocals" not in d.sources
