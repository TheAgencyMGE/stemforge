"""Score engine combinations from saved evaluate_slakh.py notes, without re-running inference.

    python scripts/rescore_slakh.py --root <BabySlakh dir> results.json

Policies compared on identical separated sources:
* each single engine as run;
* hybrid: the bass track from the policy engine (Basic Pitch on the bass stem), every other pitched
  track from YourMT3+ mix transcription attributed to sources;
and, for the piano source, Basic Pitch vs Transkun vs YourMT3+ against the true piano part.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import note_f1  # noqa: E402
from evaluate_slakh import load_track  # noqa: E402


def expand(compact: list[list[float]]) -> list[dict]:
    return [{"pitch": int(p), "start": s, "end": e, "velocity": v} for p, s, e, v in compact]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("results")
    args = ap.parse_args()
    rows: dict[str, list[dict]] = {}
    piano_rows: dict[str, list[float]] = {}
    for r in json.load(open(args.results)):
        if "error" in r:
            continue
        eng = {e["engine"]: e for e in r["transcription"] if "notes" in e}
        if "policy" not in eng or "yourmt3-mix" not in eng:
            continue
        tr = load_track(Path(args.root) / r["track"], args.start, args.seconds)
        truth = [n for s in tr["stems"] if not s["is_drum"] for n in s["notes"]]
        fam_truth = {f: [n for s in tr["stems"] if s["family"] == f for n in s["notes"]] for f in ("bass", "keyboards", "guitar")}
        piano_truth = [n for s in tr["stems"] if s["class"] == "Piano" for n in s["notes"]]
        pol = {k: expand(v) for k, v in eng["policy"]["notes"].items()}
        ymt = {k: expand(v) for k, v in eng["yourmt3-mix"]["notes"].items()}
        hybrid = {k: (pol[k] if k == "bass" else ymt.get(k, [])) for k in set(pol) | set(ymt)}
        for name, srcs in (("policy", pol), ("yourmt3-mix", ymt), ("hybrid: bass=policy, rest=yourmt3-mix", hybrid)):
            union = [n for v in srcs.values() for n in v]
            row = {"union": note_f1(truth, union)["f1"], "offsets": note_f1(truth, union, offsets=True)["f1"]}
            for f, ref in fam_truth.items():
                if ref and srcs:
                    row[f] = max(note_f1(ref, v)["f1"] or 0.0 for v in srcs.values())
            rows.setdefault(name, []).append(row)
        if piano_truth and "piano" in pol:
            for name in ("policy", "transkun", "yourmt3-mix"):
                notes = eng.get(name, {}).get("notes", {}).get("piano")
                if notes is not None:
                    piano_rows.setdefault(name, []).append(note_f1(piano_truth, expand(notes))["f1"])

    def med(vals):
        vals = [v for v in vals if v is not None]
        return f"{median(vals):.3f} (n={len(vals)})" if vals else "–"

    print("| Policy | All pitched onset F1 | Onset+offset F1 | Bass F1 | Keyboards F1 | Guitar F1 |")
    print("|---|---|---|---|---|---|")
    for name, rs in rows.items():
        print(f"| {name} | {med([r['union'] for r in rs])} | {med([r['offsets'] for r in rs])} | "
              f"{med([r.get('bass') for r in rs])} | {med([r.get('keyboards') for r in rs])} | {med([r.get('guitar') for r in rs])} |")
    print("\n| Engine on the separated piano source | F1 vs the true piano part |")
    print("|---|---|")
    for name, vals in piano_rows.items():
        print(f"| {name} | {med(vals)} |")


if __name__ == "__main__":
    main()
