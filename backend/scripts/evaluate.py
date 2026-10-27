"""Measure discovery, separation and transcription on fixtures with exact ground truth.

    python scripts/evaluate.py [--fixtures demo keys_solo drums_bass long_demo] [--out ../docs/EVALUATION.md]

The path scored is the production one: HTDemucs, mixture-silence gate, source discovery (PANNs
tags if installed), the policy engine per source, note refinement, and cross-source attribution.
Raw model notes and refined notes are both scored, so every refinement is justified by numbers.

Metrics (mir_eval): note onset F1 (50 ms, 50 cents, offsets ignored); onset+offset F1 (offset
within 20% of the note or 50 ms); drums: onset F1 per GM class. Fixtures are synthetic (see
samples/fixtures.py); the numbers validate behaviour, they do not predict accuracy on commercial
recordings.
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
sys.path.insert(0, str(ROOT.parent / "samples"))

from stemforge.pipeline import audio, refine, transcription  # noqa: E402
from stemforge.pipeline.discovery import discover  # noqa: E402
from stemforge.pipeline.separation import DemucsSeparator, gate_mixture_silence, resolve_device  # noqa: E402


def transcription_hz(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _arrays(notes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    iv = np.array([[n["start"], max(n["end"], n["start"] + 0.01)] for n in notes]).reshape(-1, 2)
    return iv, np.array([transcription_hz(n["pitch"]) for n in notes])


def note_f1(ref: list[dict], est: list[dict], offsets: bool = False) -> dict:
    import mir_eval

    if not ref:
        return {"precision": None, "recall": None, "f1": None, "n_ref": 0, "n_est": len(est)}
    if not est:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_ref": len(ref), "n_est": 0}
    ri, rp = _arrays(ref)
    ei, ep = _arrays(est)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=0.2 if offsets else None)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "n_ref": len(ref), "n_est": len(est)}


def matched(ref: list[dict], est: list[dict]) -> list[tuple[int, int]]:
    import mir_eval

    if not ref or not est:
        return []
    ri, rp = _arrays(ref)
    ei, ep = _arrays(est)
    return mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)


def onset_f1(ref: list[float], est: list[float]) -> dict:
    import mir_eval

    if not ref:
        return {"f1": None}
    f, p, r = mir_eval.onset.f_measure(np.array(sorted(ref)), np.array(sorted(est)), window=0.05)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "n_ref": len(ref), "n_est": len(est)}


def run_fixture(name: str, mix: np.ndarray, truth: dict) -> dict:
    sr = audio.TARGET_SR
    seps: dict[str, DemucsSeparator] = {}

    def separate(model, x, cb):
        seps.setdefault(model, DemucsSeparator(model))
        return gate_mixture_silence(x, seps[model].separate(x, sr, cb), sr)

    t0 = time.monotonic()
    base = separate("htdemucs", mix, lambda f: None)
    t_sep = time.monotonic() - t0
    t0 = time.monotonic()
    d = discover(mix, sr, base, "htdemucs", separate, lambda f, m: None)
    t_disc = time.monotonic() - t0

    families = {s.meta["family"] for s in d.sources.values() if s.status == "accepted"}
    expect = truth.get("expect", {})
    discovery = {
        "sources": {k: f"{s.meta['display']} ({s.status}, {s.meta.get('isolation')})" for k, s in d.sources.items()},
        "present_ok": {f: f in families for f in expect.get("present", [])},
        "absent_ok": {f: f not in families for f in expect.get("absent", [])},
        "reconstruction": d.reconstruction, "tagger": d.tagger,
    }

    raw: dict[str, list[dict]] = {}
    refined: dict[str, list[dict]] = {}
    t0 = time.monotonic()
    for stem, src in d.sources.items():
        eng, params, _ = transcription.choose_engine(stem)
        feats = transcription.compute_features(eng, src.audio, sr, params)
        raw[stem] = transcription.notes_from_features(eng, feats, src.audio, sr, params)
        refined[stem] = refine.apply(stem, eng, raw[stem], src.audio, sr).notes
    pitched = [s for s in refined if s not in ("drums", "percussion")]
    dupes = refine.cross_source_duplicates({s: refined[s] for s in pitched})
    for s, ids in dupes.items():
        refined[s] = [n for n in refined[s] if n["id"] not in set(ids)]
    t_tr = time.monotonic() - t0

    rows: list[dict] = []
    notes = truth["notes"]
    pitched_parts = {k: v for k, v in notes.items() if k != "drums"}
    for part, ref in pitched_parts.items():
        # Each reference part is scored against the source that transcribes it best.
        best = max(pitched, key=lambda s: note_f1(ref, refined[s])["f1"] or 0.0) if pitched else None
        if best is None:
            rows.append({"part": part, "source": "-", "raw": None, "refined": None})
            continue
        row = {"part": part, "source": d.sources[best].meta["display"],
               "raw": note_f1(ref, raw[best]), "refined": note_f1(ref, refined[best]),
               "refined_offsets": note_f1(ref, refined[best], offsets=True)}
        quiet = [n for n in ref if n.get("gain", 1.0) < 0.5]
        if quiet:
            row["quiet_recall"] = note_f1(quiet, refined[best])["recall"]
        pairs = matched(ref, refined[best])
        gains = [ref[i].get("gain") for i, _ in pairs]
        if pairs and all(g is not None for g in gains) and len(set(gains)) > 1:
            vel = np.array([refined[best][j]["velocity"] for _, j in pairs], dtype=float)
            g = np.array(gains, dtype=float)
            row["velocity_loud_minus_quiet"] = round(float(vel[g >= 0.5].mean() - vel[g < 0.5].mean()), 1)
        rows.append(row)
    all_ref = [n for v in pitched_parts.values() for n in v]
    union = {"raw": note_f1(all_ref, [n for s in pitched for n in raw[s]]),
             "refined": note_f1(all_ref, [n for s in pitched for n in refined[s]])} if all_ref else None
    drums = []
    if "drums" in notes:
        est = refined.get("drums", [])
        for cls, pitch in transcription.GM_DRUMS.items():
            drums.append({"class": cls, **onset_f1([n["start"] for n in notes["drums"] if n["pitch"] == pitch],
                                                   [n["start"] for n in est if n["pitch"] == pitch])})
    return {
        "fixture": name, "duration": round(mix.shape[1] / sr, 2), "device": resolve_device(),
        "seconds": {"separation": round(t_sep, 1), "discovery": round(t_disc, 1), "transcription": round(t_tr, 1)},
        "discovery": discovery, "rows": rows, "union": union, "drums": drums,
    }


def main() -> None:
    from fixtures import FIXTURES, fixture

    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="*", default=list(FIXTURES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = []
    for name in args.fixtures:
        mix, truth = fixture(name)
        print(f"== {name}", flush=True)
        results.append(run_fixture(name, mix, truth))
        print(json.dumps(results[-1], indent=1, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=float))
    if args.out:
        out = Path(args.out)
        marker = "<!-- manual:"
        tail = ""
        if out.exists() and marker in (old := out.read_text(encoding="utf-8")):
            tail = "\n" + old[old.index(marker):]
        out.write_text(render_md(results) + tail, encoding="utf-8")


def _f(m: dict | None, key: str = "f1") -> str:
    return "–" if not m or m.get(key) is None else str(m[key])


def render_md(results: list[dict]) -> str:
    lines = [
        "# Evaluation",
        "",
        "Generated by `backend/scripts/evaluate.py` on the CC0 fixtures in `samples/` (`fixtures.py`), whose "
        "ground truth is exact. The scored path is the production pipeline: HTDemucs, mixture-silence gate, "
        "source discovery, policy engine, note refinement and cross-source attribution. **Raw** is the "
        "engine output; **refined** is what the app shows. The fixtures are synthetic: they check behaviour "
        "(absence, quiet and repeated notes, silence, reverb, chunk boundaries) and do not predict accuracy "
        "on commercial recordings.",
        "",
        "## Discovery",
        "",
        "| Fixture | Duration | Sources found | Expected present | Expected absent | Mix rebuilt to | Time sep / disc / transcr (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        d = r["discovery"]
        ok = lambda m: ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in m.items()) or "–"  # noqa: E731
        lines.append(f"| {r['fixture']} | {r['duration']} s | {'; '.join(d['sources'].values())} | {ok(d['present_ok'])} | "
                     f"{ok(d['absent_ok'])} | {d['reconstruction']['residual_db']} dB | "
                     f"{r['seconds']['separation']} / {r['seconds']['discovery']} / {r['seconds']['transcription']} |")
    lines += ["", "## Pitched transcription (note onset F1)", "",
              "| Fixture | Part | Best source | Raw P / R / F1 | Refined P / R / F1 | Refined onset+offset F1 | Quiet-note recall | Velocity loud − quiet |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        for row in r["rows"]:
            raw, ref = row.get("raw"), row.get("refined")
            lines.append(
                f"| {r['fixture']} | {row['part']} | {row['source']} | {_f(raw, 'precision')} / {_f(raw, 'recall')} / **{_f(raw)}** | "
                f"{_f(ref, 'precision')} / {_f(ref, 'recall')} / **{_f(ref)}** | {_f(row.get('refined_offsets'))} | "
                f"{row.get('quiet_recall', '–')} | {row.get('velocity_loud_minus_quiet', '–')} |")
        if r.get("union"):
            u = r["union"]
            lines.append(f"| {r['fixture']} | *all pitched parts* | *all pitched sources* | {_f(u['raw'], 'precision')} / "
                         f"{_f(u['raw'], 'recall')} / **{_f(u['raw'])}** | {_f(u['refined'], 'precision')} / "
                         f"{_f(u['refined'], 'recall')} / **{_f(u['refined'])}** | | | |")
    lines += ["", "## Drums (onset F1 per GM class)", "", "| Fixture | Class | Precision | Recall | F1 | Ref | Est |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        for row in r["drums"]:
            lines.append(f"| {r['fixture']} | {row['class']} | {_f(row, 'precision')} | {_f(row, 'recall')} | "
                         f"**{_f(row)}** | {row.get('n_ref', '–')} | {row.get('n_est', '–')} |")
    lines += ["", f"Device: {results[0]['device'] if results else '–'}. Velocity column: mean MIDI velocity of matched "
              "loud notes minus quiet notes (the keys fixture's quiet bars are ~16 dB down); positive means the "
              "dynamics survived."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
