"""Does render fidelity pick the more accurate engine? Validated on BabySlakh with saved notes.

    python scripts/select_by_render.py --root <BabySlakh dir> results.json

For each pitched source, the policy (Basic Pitch) and YourMT3+ (mix, attributed) notes saved by
evaluate_slakh.py are rendered through the soundfont and compared with the source's own audio
(render.compare). The engine with the higher pitch+rhythm score is selected. The selection never
looks at ground truth; the score against ground truth is computed afterwards to judge it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import note_f1  # noqa: E402
from evaluate_slakh import SR, load_track  # noqa: E402
from rescore_slakh import expand  # noqa: E402

from stemforge.pipeline import midi, render  # noqa: E402
from stemforge.pipeline.discovery import discover  # noqa: E402
from stemforge.pipeline.separation import DemucsSeparator, gate_mixture_silence  # noqa: E402

PROGRAM = {"bass": 33, "guitar": 25, "piano": 0}


def render_score(notes: list[dict], source, program: int, seconds: float, td: Path) -> float:
    if not notes:
        return -1.0
    path = td / "x.mid"
    midi.build_midi([midi.MidiTrackSpec("x", [{**n, "velocity": n["velocity"] or 80} for n in notes], program)], 120.0).save(str(path))
    c = render.compare(render.render_midi(path, SR, seconds), source, SR)
    return (c["chroma_similarity"] or 0.0) + (c["onset_correlation"] or 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("results")
    args = ap.parse_args()
    seps: dict[str, DemucsSeparator] = {}

    def separate(model, x, cb):
        seps.setdefault(model, DemucsSeparator(model))
        return gate_mixture_silence(x, seps[model].separate(x, SR, cb), SR)

    rows = {"policy": [], "yourmt3-mix": [], "selected by render fidelity": [], "oracle choice (best possible)": []}
    picks = {"policy": 0, "yourmt3-mix": 0}
    agree = total = 0
    for r in json.load(open(args.results)):
        eng = {e["engine"]: e for e in r.get("transcription", []) if "notes" in e}
        if "policy" not in eng or "yourmt3-mix" not in eng:
            continue
        tr = load_track(Path(args.root) / r["track"], 60.0, 30.0)
        d = discover(tr["mix"], SR, separate("htdemucs", tr["mix"], lambda f: None), "htdemucs", separate,
                     lambda f, m: None)
        fam_ref = {f: [n for s in tr["stems"] if s["family"] == f for n in s["notes"]] for f in ("bass", "keyboards", "guitar")}
        with tempfile.TemporaryDirectory() as td:
            for name, src in d.sources.items():
                if name in ("drums", "percussion"):
                    continue
                cand = {e: expand(eng[e]["notes"].get(name, [])) for e in ("policy", "yourmt3-mix")}
                score = {e: render_score(n, src.audio, PROGRAM.get(name, 0), 30.0, Path(td)) for e, n in cand.items()}
                choice = max(score, key=lambda e: score[e])
                picks[choice] += 1
                fam = {"bass": "bass", "guitar": "guitar"}.get(name, "keyboards")
                ref = fam_ref[fam]
                if not ref:
                    continue
                f1 = {e: note_f1(ref, n)["f1"] or 0.0 for e, n in cand.items()}
                for e in ("policy", "yourmt3-mix"):
                    rows[e].append(f1[e])
                rows["selected by render fidelity"].append(f1[choice])
                rows["oracle choice (best possible)"].append(max(f1.values()))
                total += 1
                agree += int(f1[choice] >= max(f1.values()) - 1e-9)
        print(r["track"], "done", flush=True)
    print("\n| Per-source choice | Median F1 vs the true part | Mean F1 | Sources |")
    print("|---|---|---|---|")
    for k, v in rows.items():
        print(f"| {k} | {median(v):.3f} | {sum(v) / len(v):.3f} | {len(v)} |")
    print(f"\nRender fidelity picked the better engine for {agree} of {total} sources. Picks: {picks}.")


if __name__ == "__main__":
    main()
