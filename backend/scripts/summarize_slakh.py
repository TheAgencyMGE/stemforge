"""Turn evaluate_slakh.py result files into the benchmark tables in docs/EVALUATION.md.

    python scripts/summarize_slakh.py results1.json [results2.json ...] > table.md

Engines from several files are merged per track (the separation stage is deterministic, so every
engine saw the same discovered sources). Medians are over tracks where the metric exists; "wins"
counts tracks where an engine beats the policy engine on the same track.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median


def load(paths: list[str]) -> dict[str, dict]:
    tracks: dict[str, dict] = {}
    for path in paths:
        for r in json.load(open(path)):
            if "error" in r:
                continue
            t = tracks.setdefault(r["track"], {k: v for k, v in r.items() if k != "transcription"} | {"engines": {}})
            for e in r["transcription"]:
                if "unavailable" not in e:
                    t["engines"][e["engine"]] = e
    return tracks


def med(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    return f"{median(vals):.3f} (n={len(vals)})" if vals else "–"


def main() -> None:
    tracks = load(sys.argv[1:])
    out = []
    out.append(f"BabySlakh tracks scored: {len(tracks)} (30 s window from 60 s, 16 kHz source audio).\n")

    # Discovery
    truth_f = defaultdict(int)
    found_f = defaultdict(int)
    hit_f = defaultdict(int)
    false_vocals = 0
    for t in tracks.values():
        tf, ff = set(t["truth_families"]), set(t["found_families"])
        # A vocal track is fabricated only where no voice-type part (GM 52-54, e.g. Choir Aahs) plays.
        false_vocals += int(t["fabricated_vocals"] and "vocals" not in tf)
        for f in tf:
            truth_f[f] += 1
            hit_f[f] += int(f in ff)
        for f in ff:
            found_f[f] += 1
    out.append("### Discovery: instrument families present vs found\n")
    out.append("| Family | Tracks where present | Found when present | Reported but absent |")
    out.append("|---|---|---|---|")
    for f in sorted(set(truth_f) | set(found_f)):
        out.append(f"| {f} | {truth_f[f]} | {hit_f[f]} | {found_f[f] - hit_f[f]} |")
    with_voice = sum(1 for t in tracks.values() if "vocals" in t["truth_families"])
    out.append(f"\nVocal tracks reported where no voice-type part plays: **{false_vocals}** of {len(tracks) - with_voice} "
               f"songs (BabySlakh has no sung vocals; {with_voice} songs contain sampled voice pads, GM 52-54).\n")

    # Separation
    out.append("### Separation (SI-SDR of our source vs the true stems of that family, dB)\n")
    out.append("| Source | Median | Tracks where extracted |")
    out.append("|---|---|---|")
    for src in ("drums", "bass", "guitar", "piano"):
        vals = [t["si_sdr"].get(src) for t in tracks.values()]
        out.append(f"| {src} | {med(vals)} | {sum(v is not None for v in vals)} |")
    out.append(f"\nMixture rebuilt from all sources: median {med([t['reconstruction']['residual_db'] for t in tracks.values()])} dB.\n")

    # Transcription
    engines = sorted({e for t in tracks.values() for e in t["engines"]}, key=lambda e: (e != "policy", e))
    out.append("### Transcription on separated sources (production path)\n")
    out.append("| Engine | All pitched notes, onset F1 | Onset+offset F1 | Bass F1 | Keyboards F1 | Guitar F1 | Tracks better than policy (all-notes F1) |")
    out.append("|---|---|---|---|---|---|---|")
    for e in engines:
        rows = [t["engines"][e] for t in tracks.values() if e in t["engines"]]
        wins = sum(1 for t in tracks.values() if e in t["engines"] and "policy" in t["engines"]
                   and t["engines"][e]["separated_union"]["f1"] > t["engines"]["policy"]["separated_union"]["f1"])
        same = [t for t in tracks.values() if e in t["engines"] and "policy" in t["engines"]]
        fam = {f: med([r["separated_by_family"].get(f, {}).get("f1") for r in rows]) for f in ("bass", "keyboards", "guitar")}
        out.append(f"| {e} | {med([r['separated_union']['f1'] for r in rows])} | "
                   f"{med([r['separated_union_offsets']['f1'] for r in rows])} | {fam['bass']} | {fam['keyboards']} | "
                   f"{fam['guitar']} | {'–' if e == 'policy' else f'{wins} of {len(same)}'} |")

    out.append("\n### Transcription on the true isolated stems (oracle: no separation error)\n")
    out.append("| Engine | Instrument class | Note F1 | Onset+offset F1 | Velocity r | Stems |")
    out.append("|---|---|---|---|---|---|")
    for e in engines:
        by_class: dict[str, list[dict]] = defaultdict(list)
        for t in tracks.values():
            for o in t["engines"].get(e, {}).get("oracle", []):
                by_class[o["class"]].append(o)
        for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
            os_ = by_class[cls]
            if len(os_) < 2:
                continue
            out.append(f"| {e} | {cls} | {med([o['f1'] for o in os_])} | {med([o['offsets_f1'] for o in os_])} | "
                       f"{med([o.get('velocity_r') for o in os_])} | {len(os_)} |")

    renders = [(t["track"], o) for t in tracks.values() for o in t["engines"].get("policy", {}).get("oracle", [])
               if o.get("render")]
    if renders:
        out.append("\n### MIDI render fidelity (oracle stems, Basic Pitch policy, MuseScore General soundfont)\n")
        out.append("| Comparison | Pitch match (chroma) | Rhythm (onset corr.) | Sound diff. (dB) |")
        out.append("|---|---|---|---|")
        for key, label in (("note_error", "render(predicted MIDI) vs render(true MIDI): note errors only"),
                           ("instrument_gap", "render(true MIDI) vs true audio: soundfont vs real instrument"),
                           ("total", "render(predicted MIDI) vs true audio: both")):
            out.append(f"| {label} | {med([o['render'][key]['chroma_similarity'] for _, o in renders])} | "
                       f"{med([o['render'][key]['onset_correlation'] for _, o in renders])} | "
                       f"{med([o['render'][key]['logmel_l1_db'] for _, o in renders])} |")

    drums = [t["drums"] for t in tracks.values() if t.get("drums")]
    if drums:
        out.append("\n### Drums (our drum source, onset F1 per class)\n")
        out.append("| Class | Median F1 |")
        out.append("|---|---|")
        for c in ("kick", "snare", "hihat"):
            out.append(f"| {c} | {med([d[c].get('f1') for d in drums])} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
