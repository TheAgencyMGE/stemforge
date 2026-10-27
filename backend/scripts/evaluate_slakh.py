"""Benchmark discovery, separation, transcription and MIDI render fidelity on BabySlakh.

BabySlakh (Manilow et al., WASPAA 2019; https://zenodo.org/records/4603870, CC BY 4.0) is 20
multi-instrument songs rendered from MIDI with professional sample libraries, with every
instrument's audio and MIDI. Download and unpack it, then:

    python scripts/evaluate_slakh.py --root <dir containing Track00001...> [--tracks 20]
        [--start 60 --seconds 30] [--engines basic-pitch transkun muscriptor] [--json out.json]

The audio is 16 kHz (resampled to 44.1 kHz for the pipeline), so separation is judged without
content above 8 kHz. Every engine sees identical inputs:

* **separated**: the production path (HTDemucs, discovery, policy engine, refinement);
* **oracle**: the true isolated stem of each instrument, which removes separation error and leaves
  only the transcriber's.

Metrics are mir_eval note onset F1 (50 ms), onset+offset F1, drum onset F1 per class, SI-SDR of
separated stems, and render fidelity (render.py) where FluidSynth is configured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import librosa  # noqa: E402
from evaluate import note_f1, onset_f1  # noqa: E402

from stemforge.pipeline import audio, external, refine, render, transcription  # noqa: E402
from stemforge.pipeline.attribution import attribute_notes  # noqa: E402
from stemforge.pipeline.discovery import discover  # noqa: E402
from stemforge.pipeline.separation import DemucsSeparator, gate_mixture_silence  # noqa: E402
from stemforge.pipeline.transcription import _velocity_curve  # noqa: E402

SR = audio.TARGET_SR


def gm_family(program: int, is_drum: bool) -> str:
    """Family from the General MIDI program, which is what the stem actually plays. Slakh's
    inst_class is coarser: "Strings (continued)" also holds synth strings, Choir Aahs and
    orchestra hits (programs 50-55)."""
    if is_drum:
        return "drums"
    p = int(program)
    if p == 46:
        return "plucked"
    if p == 47:
        return "percussion"
    if p in (21, 22):
        return "other"
    for lo, hi, fam in ((0, 23, "keyboards"), (24, 31, "guitar"), (32, 39, "bass"), (40, 51, "strings"),
                        (52, 54, "vocals"), (55, 55, "other"), (56, 63, "brass"), (64, 79, "woodwinds"),
                        (80, 95, "keyboards"), (96, 103, "other"), (104, 111, "plucked"), (112, 119, "percussion")):
        if lo <= p <= hi:
            return fam
    return "other"


DRUM_CLASS = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 39: "snare", 40: "snare",
              42: "hihat", 44: "hihat", 46: "hihat"}


def load_track(path: Path, start: float, seconds: float) -> dict:
    import pretty_midi
    import soundfile as sf
    import soxr
    import yaml

    meta = yaml.safe_load((path / "metadata.yaml").read_text())
    mix16, sr16 = sf.read(str(path / "mix.wav"), dtype="float32", always_2d=True)
    i0, i1 = int(start * sr16), int((start + seconds) * sr16)

    def up(x: np.ndarray) -> np.ndarray:
        x = x[i0:i1]
        x = np.pad(x, ((0, max(0, (i1 - i0) - len(x))), (0, 0)))
        y = soxr.resample(x, sr16, SR).T
        return np.ascontiguousarray(np.repeat(y, 2, axis=0) if y.shape[0] == 1 else y, dtype=np.float32)

    stems = []
    for sid, info in meta["stems"].items():
        wav, mid = path / "stems" / f"{sid}.wav", path / "MIDI" / f"{sid}.mid"
        if not (wav.exists() and mid.exists()):
            continue
        x, _ = sf.read(str(wav), dtype="float32", always_2d=True)
        a = up(x)
        pm = pretty_midi.PrettyMIDI(str(mid))
        notes = [{"pitch": n.pitch, "start": round(n.start - start, 4), "end": round(min(n.end, start + seconds) - start, 4),
                  "velocity": n.velocity}
                 for inst in pm.instruments for n in inst.notes if start <= n.start < start + seconds]
        level = float(np.sqrt(np.mean(a ** 2)))
        if not notes or level < 1e-4:
            continue
        # Only bass is corrected: there the offset was confirmed three ways (pYIN on three songs,
        # YourMT3+ F1 0.99 after the shift, library transposition). In polyphonic stems pYIN locks
        # onto low notes and would wrongly move parts that transcribers already match unshifted.
        fam = gm_family(info["program_num"], info["is_drum"])
        shift = sounding_offset(a, notes) if fam == "bass" and _monophonic(notes) else 0
        if shift:
            notes = [{**n, "pitch": n["pitch"] + shift} for n in notes]
        stems.append({"id": sid, "class": info["inst_class"], "family": gm_family(info["program_num"], info["is_drum"]),
                      "octave_shift": shift,
                      "program": int(info["program_num"]), "is_drum": bool(info["is_drum"]),
                      "audio": a, "notes": notes, "level": level})
    return {"name": path.name, "mix": up(mix16), "stems": stems}


def sounding_offset(x: np.ndarray, notes: list[dict]) -> int:
    """Octave transposition between a stem's written MIDI and what it sounds like.

    Some BabySlakh sample libraries sound an octave below their MIDI (all three bass stems checked:
    written median 52-55, sounding 40-43). Transcribers report sounding pitch, which is also what a
    General MIDI instrument needs, so references are moved to sounding pitch. Measured with pYIN
    (fundamental, not the strongest harmonic); applied only for a clean multiple of 12."""
    y = librosa.resample(audio.to_mono(x), orig_sr=SR, target_sr=22050, res_type="soxr_hq")
    f0, voiced, _ = librosa.pyin(y, fmin=30, fmax=1500, sr=22050, frame_length=4096)
    if voiced.mean() < 0.2:
        return 0
    diff = float(np.nanmedian(librosa.hz_to_midi(f0[voiced]))) - float(np.median([n["pitch"] for n in notes]))
    k = int(round(diff / 12)) * 12
    return k if k != 0 and abs(diff - k) < 1.5 else 0


def _monophonic(notes: list[dict]) -> bool:
    ev = sorted(notes, key=lambda n: n["start"])
    overlaps = sum(1 for a, b in zip(ev, ev[1:], strict=False) if b["start"] < a["end"] - 0.03)
    return overlaps <= 0.1 * max(1, len(ev))


def compact(notes: list[dict]) -> list[list[float]]:
    return [[int(n["pitch"]), round(float(n["start"]), 4), round(float(n["end"]), 4), int(n.get("velocity") or 0)]
            for n in notes]


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float | None:
    e, r = audio.to_mono(est).astype(np.float64), audio.to_mono(ref).astype(np.float64)
    if np.sum(r ** 2) < 1e-9:
        return None
    a = np.dot(e, r) / np.dot(r, r)
    t = a * r
    return round(float(10 * np.log10(np.sum(t ** 2) / max(np.sum((e - t) ** 2), 1e-12))), 2)


def transcribe(engine: str, x: np.ndarray, stem: str, identity: str | None = None) -> list[dict]:
    if engine == "yourmt3":
        return [n for n in external.yourmt3_notes(x, SR) if not n["is_drum"]]
    if engine == "transkun":
        return external.transkun_notes(x, SR)
    if engine == "muscriptor":
        groups = external.REGISTRY_TO_MUSCRIPTOR.get(identity or "", None)
        return external.muscriptor_notes(x, SR, groups)
    eng, params, _ = transcription.choose_engine(stem)
    if engine != "policy":
        eng, params = engine, transcription.resolve_params(engine, {})
    feats = transcription.compute_features(eng, x, SR, params)
    return transcription.notes_from_features(eng, feats, x, SR, params)


def velocity_error(ref: list[dict], est: list[dict]) -> dict:
    """Velocity agreement on matched notes: Pearson r and mean absolute error after fitting a linear
    map (velocity scales differ between instruments and engines, so only the shape is judged)."""
    from evaluate import matched

    pairs = matched(ref, est)
    if len(pairs) < 8:
        return {"velocity_r": None, "velocity_mae": None}
    v_ref = np.array([ref[i]["velocity"] for i, _ in pairs], dtype=float)
    v_est = np.array([est[j]["velocity"] for _, j in pairs], dtype=float)
    if v_ref.std() == 0 or v_est.std() == 0:
        return {"velocity_r": None, "velocity_mae": None}
    a, b = np.polyfit(v_est, v_ref, 1)
    return {"velocity_r": round(float(np.corrcoef(v_ref, v_est)[0, 1]), 3),
            "velocity_mae": round(float(np.abs(a * v_est + b - v_ref).mean()), 1)}


def render_fidelity(pred: list[dict], ref: list[dict], program: int, is_drum: bool, truth_audio: np.ndarray,
                    seconds: float, td: Path) -> dict | None:
    if render.renderer() is None or not pred:
        return None
    from stemforge.pipeline import midi as midi_mod

    def write(notes, name):
        spec = [midi_mod.MidiTrackSpec(name, [{**n, "velocity": n.get("velocity", 90)} for n in notes], program, is_drum)]
        path = td / f"{name}.mid"
        midi_mod.build_midi(spec, 120.0).save(str(path))
        return render.render_midi(path, SR, seconds)

    r_pred, r_ref = write(pred, "pred"), write(ref, "ref")
    return {"note_error": render.compare(r_pred, r_ref, SR), "instrument_gap": render.compare(r_ref, truth_audio, SR),
            "total": render.compare(r_pred, truth_audio, SR)}


def evaluate_track(tr: dict, engines: list[str], seconds: float, td: Path, oracle_engines: set[str] | None = None) -> dict:
    mix = tr["mix"]
    seps: dict[str, DemucsSeparator] = {}

    def separate(model, x, cb):
        seps.setdefault(model, DemucsSeparator(model))
        return gate_mixture_silence(x, seps[model].separate(x, SR, cb), SR)

    t0 = time.monotonic()
    base = separate("htdemucs", mix, lambda f: None)
    d = discover(mix, SR, base, "htdemucs", separate, lambda f, m: None)
    t_sep = time.monotonic() - t0

    truth_fams = {s["family"] for s in tr["stems"]}
    found: set[str] = set()
    for s in d.sources.values():
        found.add(s.meta["family"])
        found.update(c for c in s.meta.get("families", []) or [])
        if s.meta.get("kind") == "group":
            from stemforge.pipeline.instruments import INSTRUMENTS

            found.update(INSTRUMENTS[c["id"]].family for c in s.meta.get("contains", []) if c["id"] in INSTRUMENTS)
    # Tag evidence per family on the mix and on the final residual, for threshold calibration.
    from stemforge.pipeline.discovery import FAMILY_LABEL, Evidence, family_score

    ev_mix = Evidence(mix, mix, SR, True)
    ev_res = Evidence(d.sources["other"].audio, mix, SR, True) if "other" in d.sources else None
    fam_scores = {f: {"mix": round(family_score(ev_mix, f), 4),
                      "residual": round(family_score(ev_res, f), 4) if ev_res is not None else None}
                  for f in FAMILY_LABEL}
    res: dict = {"track": tr["name"], "truth_families": sorted(truth_fams), "found_families": sorted(found),
                 "octave_shifts": {s["id"]: s["octave_shift"] for s in tr["stems"] if s["octave_shift"]},
                 "family_scores": fam_scores,
                 "sources": {k: s.meta["display"] for k, s in d.sources.items()},
                 "fabricated_vocals": "vocals" in d.sources, "reconstruction": d.reconstruction,
                 "seconds_separation": round(t_sep, 1)}

    # Separation: our source vs the sum of the true stems of that family.
    fam_audio: dict[str, np.ndarray] = {}
    for s in tr["stems"]:
        fam_audio[s["family"]] = fam_audio.get(s["family"], 0) + s["audio"]
    sep = {}
    for ours, fam in (("drums", "drums"), ("bass", "bass"), ("guitar", "guitar"), ("piano", "keyboards")):
        if ours in d.sources and fam in fam_audio:
            sep[ours] = si_sdr(d.sources[ours].audio, fam_audio[fam])
    res["si_sdr"] = sep

    # Transcription on separated sources (production path) and on true stems (oracle).
    rows = []
    pitched_truth = [n for s in tr["stems"] if not s["is_drum"] for n in s["notes"]]
    for engine in engines:
        try:
            t0 = time.monotonic()
            sep_notes: dict[str, list[dict]] = {}
            if engine == "yourmt3-mix":
                # Mix-level transcription; separated sources only decide which track a note belongs
                # to and supply its velocity (YourMT3+ emits none).
                mix_notes = [n for n in external.yourmt3_notes(mix, SR) if not n["is_drum"]]
                pitched_sources = {k: s.audio for k, s in d.sources.items() if k not in ("drums", "percussion")}
                sep_notes = attribute_notes(mix_notes, pitched_sources, SR)
                for k, ns in sep_notes.items():
                    curve = _velocity_curve(librosa.resample(audio.to_mono(pitched_sources[k]), orig_sr=SR,
                                                             target_sr=22050), 22050)
                    for n in ns:
                        n["velocity"] = curve(n["start"], n["end"])
            for name, src in ([] if engine == "yourmt3-mix" else d.sources.items()):
                if name in ("drums", "percussion"):
                    continue
                if engine == "transkun" and name != "piano":
                    continue
                raw = transcribe(engine, src.audio, name, src.meta.get("identity"))
                eng_id = transcription.choose_engine(name)[0] if engine == "policy" else engine.split("-")[0]
                sep_notes[name] = refine.apply(name, eng_id, raw, src.audio, SR).notes
            dupes = refine.cross_source_duplicates(sep_notes)
            for s_, ids in dupes.items():
                sep_notes[s_] = [n for n in sep_notes[s_] if n["id"] not in set(ids)]
            union = [n for v in sep_notes.values() for n in v]
            row = {"engine": engine, "notes": {k: compact(v) for k, v in sep_notes.items()},
                   "separated_union": note_f1(pitched_truth, union),
                   "separated_union_offsets": note_f1(pitched_truth, union, offsets=True),
                   "seconds": None}
            per = {}
            for fam in ("bass", "keyboards", "guitar"):
                ref = [n for s in tr["stems"] if s["family"] == fam for n in s["notes"]]
                if ref and sep_notes:
                    best = max(sep_notes, key=lambda k: note_f1(ref, sep_notes[k])["f1"] or 0.0)
                    per[fam] = {"source": best, **note_f1(ref, sep_notes[best])}
            row["separated_by_family"] = per
            # Oracle: each true pitched stem transcribed on its own.
            oracle = []
            for s in ([] if oracle_engines is not None and engine not in oracle_engines else tr["stems"]):
                if s["is_drum"] or (engine == "transkun" and s["class"] != "Piano"):
                    continue
                stem_name = {"bass": "bass", "guitar": "guitar"}.get(s["family"], "piano" if s["class"] == "Piano" else "other")
                est = transcribe(engine, s["audio"], stem_name, {"Piano": "piano", "Bass": "bass"}.get(s["class"]))
                o = {"stem": s["id"], "class": s["class"], "notes": compact(est), **note_f1(s["notes"], est),
                     "offsets_f1": note_f1(s["notes"], est, offsets=True)["f1"], **velocity_error(s["notes"], est)}
                if s["class"] in ("Piano", "Bass"):
                    fid = render_fidelity(est, s["notes"], s["program"], False, s["audio"], seconds, td)
                    if fid:
                        o["render"] = fid
                oracle.append(o)
            row["oracle"] = oracle
            row["seconds"] = round(time.monotonic() - t0, 1)
            rows.append(row)
        except external.ExternalEngineError as exc:
            rows.append({"engine": engine, "unavailable": str(exc)})
    res["transcription"] = rows

    # Drums: our drum source vs the true drum notes, per class.
    drum_truth = [n for s in tr["stems"] if s["is_drum"] for n in s["notes"]]
    if drum_truth and "drums" in d.sources:
        est = transcribe("policy", d.sources["drums"].audio, "drums")
        res["drums"] = {c: onset_f1([n["start"] for n in drum_truth if DRUM_CLASS.get(n["pitch"]) == c],
                                    [n["start"] for n in est if DRUM_CLASS.get(n["pitch"]) == c])
                        for c in ("kick", "snare", "hihat")}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks", type=int, default=20)
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--engines", nargs="*", default=["policy"])
    ap.add_argument("--oracle-engines", nargs="*", default=None,
                    help="engines to also run on the true stems (default: all); yourmt3 is slow on CPU")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    import tempfile

    out = Path(args.json)
    results = json.loads(out.read_text()) if out.exists() else []
    done = {r["track"] for r in results}
    for path in sorted(Path(args.root).glob("Track*"))[: args.tracks]:
        if path.name in done:
            continue
        print(f"== {path.name}", flush=True)
        try:
            tr = load_track(path, args.start, args.seconds)
            with tempfile.TemporaryDirectory(prefix="sf-slakh-") as td:
                results.append(evaluate_track(tr, args.engines, args.seconds, Path(td),
                                              set(args.oracle_engines) if args.oracle_engines is not None else None))
        except Exception:  # noqa: BLE001 - keep going, record the failure
            results.append({"track": path.name, "error": traceback.format_exc()[-1500:]})
        out.write_text(json.dumps(results, indent=1, default=float))
        print(json.dumps(results[-1], default=float)[:600], flush=True)


if __name__ == "__main__":
    main()
