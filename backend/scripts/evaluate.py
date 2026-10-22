"""Measure separation + transcription quality on the bundled demo against its ground truth.

    python scripts/evaluate.py [--engines all] [--out ../docs/EVALUATION.md]

Note-level metrics use mir_eval (onset tolerance 50 ms, offsets ignored). Drums are scored per
GM class on onsets. Numbers are for a synthetic fixture: they validate the pipeline and the
engine policy, and are not a claim about accuracy on commercial recordings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stemforge.pipeline import analysis, audio, transcription  # noqa: E402
from stemforge.pipeline.separation import DemucsSeparator, residual_db  # noqa: E402


def note_f1(ref: list[dict], est: list[dict]) -> dict:
    import mir_eval

    if not ref:
        return {"precision": None, "recall": None, "f1": None, "n_ref": 0, "n_est": len(est)}
    if not est:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_ref": len(ref), "n_est": 0}
    ri = np.array([[n["start"], n["end"]] for n in ref])
    rp = np.array([transcription_hz(n["pitch"]) for n in ref])
    ei = np.array([[n["start"], max(n["end"], n["start"] + 0.01)] for n in est])
    ep = np.array([transcription_hz(n["pitch"]) for n in est])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=0.05,
                                                                   offset_ratio=None)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "n_ref": len(ref), "n_est": len(est)}


def transcription_hz(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def onset_f1(ref: list[float], est: list[float]) -> dict:
    import mir_eval

    if not ref:
        return {"f1": None}
    f, p, r = mir_eval.onset.f_measure(np.array(sorted(ref)), np.array(sorted(est)), window=0.05)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "n_ref": len(ref), "n_est": len(est)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(ROOT.parent / "samples" / "stemforge-demo.flac"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    truth = json.loads(Path(args.sample).with_suffix(".truth.json").read_text(encoding="utf-8"))
    mix = audio.decode(Path(args.sample), "flac")

    t0 = time.monotonic()
    ana = analysis.analyze(mix, audio.TARGET_SR)
    t_ana = time.monotonic() - t0
    sep = DemucsSeparator("htdemucs")
    t0 = time.monotonic()
    stems = sep.separate(mix, audio.TARGET_SR, lambda f: None)
    t_sep = time.monotonic() - t0
    resid = residual_db(mix, stems)

    rows = []
    notes = truth["notes"]
    # Demucs routes the synthetic lead into "other" (the vocal stem comes out near-silent), so the
    # "other" stem is scored against keys + lead.
    targets = [("drums", "drums", notes["drums"]), ("bass", "bass", notes["bass"]),
               ("keys+lead", "other", notes["keys"] + notes["lead"]), ("lead", "vocals", notes["lead"])]
    stem_levels = {s: audio.level_stats(v)["rms_db"] for s, v in stems.items()}
    for part, stem, ref in targets:
        if stem_levels[stem] < -55:
            rows.append({"part": part, "stem": stem, "engine": "-", "policy": True, "precision": None,
                         "recall": None, "f1": None, "n_ref": len(ref), "n_est": 0,
                         "seconds": 0, "note": f"stem silent ({stem_levels[stem]} dB): skipped"})
            continue
        engines = ["drum-nmf"] if stem == "drums" else ["basic-pitch", "pyin"]
        for eng in engines:
            # The policy engine gets its policy params; the alternative gets the same frequency band.
            pol_engine, pol_params, _ = transcription.choose_engine(stem)
            band = {k: pol_params[k] for k in ("min_freq", "max_freq") if k in pol_params}
            params = transcription.resolve_params(eng, pol_params if pol_engine == eng else band)
            t0 = time.monotonic()
            feats = transcription.compute_features(eng, stems[stem], audio.TARGET_SR, params)
            est = transcription.notes_from_features(eng, feats, stems[stem], audio.TARGET_SR, params)
            dt = time.monotonic() - t0
            policy = transcription.choose_engine(stem)[0] == eng
            if stem == "drums":
                for cls, pitch in transcription.GM_DRUMS.items():
                    m = onset_f1([n["start"] for n in ref if n["pitch"] == pitch],
                                 [n["start"] for n in est if n["pitch"] == pitch])
                    rows.append({"part": f"drums/{cls}", "stem": stem, "engine": eng, "policy": policy,
                                 **m, "seconds": round(dt, 2)})
            else:
                m = note_f1(ref, est)
                conf = [n["confidence"] for n in est]
                rows.append({"part": part, "stem": stem, "engine": eng, "policy": policy, **m,
                             "mean_conf": round(float(np.mean(conf)), 3) if conf else None, "seconds": round(dt, 2)})

    result = {
        "sample": Path(args.sample).name, "duration": round(mix.shape[1] / audio.TARGET_SR, 2),
        "device": __import__("stemforge.pipeline.separation", fromlist=["x"]).resolve_device(),
        "tempo_est": ana.get("tempo"), "tempo_true": truth["bpm"], "key_est": (ana.get("key") or {}).get("label"),
        "key_true": truth["key"], "analysis_seconds": round(t_ana, 2), "separation_seconds": round(t_sep, 2),
        "mix_residual_db": resid, "stem_rms_db": stem_levels, "rows": rows,
    }
    print(json.dumps(result, indent=1))
    if args.out:
        out = Path(args.out)
        marker = "<!-- manual:"
        tail = ""
        if out.exists() and marker in (old := out.read_text(encoding="utf-8")):
            tail = "\n" + old[old.index(marker):]
        out.write_text(render_md(result) + tail, encoding="utf-8")


def render_md(r: dict) -> str:
    lines = [
        "# Evaluation on the bundled demo",
        "",
        "Generated by `backend/scripts/evaluate.py`. The fixture is the synthetic CC0 song in "
        "`samples/`, whose ground-truth notes are known exactly. These numbers check that the pipeline "
        "and the engine policy work. They do **not** predict accuracy on real, mastered recordings, "
        "where separation bleed is much worse.",
        "",
        f"* Duration: {r['duration']} s. Device: **{r['device']}**.",
        f"* Separation (htdemucs) took {r['separation_seconds']} s. Mix residual |mix - Σstems| is {r['mix_residual_db']} dB.",
        f"* Tempo estimated {r['tempo_est']} BPM (true {r['tempo_true']}). Key estimated {r['key_est']} (true {r['key_true']}).",
        "",
        "| Part | Stem | Engine | Policy pick | Precision | Recall | F1 | Ref | Est | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in r["rows"]:
        lines.append(
            f"| {row['part']} | {row['stem']} | {row['engine']} | {'yes' if row['policy'] else ''} | "
            f"{row.get('precision')} | {row.get('recall')} | **{row.get('f1')}** | {row.get('n_ref')} | "
            f"{row.get('n_est')} | {row.get('seconds')} |" + (f" {row['note']}" if row.get("note") else "")
        )
    lines += ["", "Metric: mir_eval note onset F1 (50 ms tolerance, pitch within 50 cents, offsets ignored). "
              "For drums: onset F1 per GM class."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
