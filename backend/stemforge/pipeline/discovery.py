"""Source discovery: which instruments are actually in the recording, and which can be isolated.

The pipeline does not assume a stem count. It starts from a broad-family separation (HTDemucs),
then for every candidate source it asks two independent questions:

1. **Is something audible there?** Frame activity relative to the mixture (``activity.py``).
2. **What is it?** PANNs AudioSet tags computed on the candidate itself, in its active windows.

A candidate with no evidence produces no track. Its audio (usually bleed) is reassigned to the
accepted source whose tag profile it matches, or kept in a hidden internal residual so the
reconstruction diagnostics stay honest.

The "Other" stem is not shown as a bucket. It is treated as a residual to be explained:
targeted extractors run on it only when tags justify them, each extraction is a soft mask of its
parent (so target + complement == parent, sample for sample), and an extraction is kept only if
the target's own tags confirm the instrument. Whatever remains is named after the instrument(s)
the tags support, or reported as an unresolved source for review.

The tagger and the separators are evidence, not ground truth. Every decision is recorded with
its numbers in ``Discovery.log`` and per-source ``meta`` so the UI can show why.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import tagger
from .activity import source_activity
from .instruments import INSTRUMENTS, get_instrument

log = logging.getLogger(__name__)

DISCOVERY_VERSION = 1

# AudioSet class indices (class_labels_indices.csv) grouped by Stemforge instrument id. Generic
# classes ("Music", "Musical instrument") are deliberately absent: they say nothing about identity.
AUDIOSET: dict[str, tuple[int, ...]] = {
    "lead_vocal": (27, 32, 33, 34, 35, 29, 37),
    "choir": (28, 30, 31),
    "rap_vocal": (36,),
    "spoken_voice": (0, 1, 2, 3, 5),
    "drum_kit": (162, 164, 165, 166, 167, 168, 171, 172),
    "drum_machine": (163,),
    "beatbox": (218,),
    "timpani": (169,),
    "tabla": (170,),
    "wood_block": (173,),
    "tambourine": (174,),
    "shaker": (175, 176),
    "gong": (177,),
    "bass": (142,),
    "double_bass_orch": (194,),
    "guitar": (140,),
    "electric_guitar": (141,),
    "acoustic_guitar": (143, 146),
    "steel_guitar": (144,),
    "banjo": (147,),
    "sitar": (148,),
    "mandolin": (149,),
    "zither": (150,),
    "ukulele": (151,),
    "piano": (153,),
    "electric_piano": (154,),
    "organ": (155, 156, 157),
    "synthesizer": (158,),
    "sampler": (159,),
    "harpsichord": (160,),
    "mallet": (179, 180, 181, 182),
    "steelpan": (183,),
    "bells": (178, 200, 201, 202, 205, 206),
    "brass_section": (185,),
    "french_horn": (186,),
    "trumpet": (187,),
    "trombone": (188,),
    "string_ensemble": (189, 190, 192),
    "violin": (191,),
    "cello": (193,),
    "woodwind_section": (195,),
    "flute": (196,),
    "saxophone": (197,),
    "clarinet": (198,),
    "harp": (199,),
    "harmonica": (208,),
    "accordion": (209,),
    "bagpipes": (210,),
    "didgeridoo": (211,),
    "theremin": (213,),
    "singing_bowl": (214,),
    "turntable": (215,),
    "orchestra": (184,),
}
PERCUSSION_IDS = ("tambourine", "shaker", "wood_block", "gong", "tabla", "timpani", "bells")
GENERIC_PERCUSSION = (161,)

# Which instrument families a broad HTDemucs stem is expected to hold.
STEM_FAMILY = {"vocals": "vocals", "drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "keyboards"}


@dataclass
class Thresholds:
    """Decision thresholds. Defaults were set on the bundled fixture and the evaluation clips
    listed in docs/EVALUATION.md; they are configuration, not constants of nature."""

    min_active_seconds: float = 0.75
    min_rel_db: float = -30.0  # median level in active frames, relative to the mixture
    strong_active_seconds: float = 2.0
    strong_rel_db: float = -18.0
    family_tag: float = 0.15  # stem's own tag score needed to confirm its family
    # AudioSet's speech classes also fire on formant-like instruments (a "Choir Aahs" pad scored
    # 0.26 speech, 0.03 singing on BabySlakh Track00009), so speech alone needs much more evidence.
    speech_tag: float = 0.4
    identity_tag: float = 0.12  # minimum score to name an instrument
    identity_margin: float = 2.0  # top / runner-up ratio to give a single name
    trigger_tag: float = 0.08  # residual score that justifies trying a targeted extractor
    accept_target_tag: float = 0.2  # target's own score needed to keep an extraction
    reassign_similarity: float = 0.6
    residual_floor_db: float = -35.0  # stop when the residual is this far below the mixture
    max_iterations: int = 4


@dataclass
class Source:
    stem: str
    audio: np.ndarray
    status: str  # "accepted" | "unresolved"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Discovery:
    sources: dict[str, Source]
    internal_residual: np.ndarray | None
    log: list[dict[str, Any]]
    reconstruction: dict[str, Any]
    tagger: str  # model name, or "unavailable"

    def summary(self) -> dict[str, Any]:
        return {
            "version": DISCOVERY_VERSION, "tagger": self.tagger, "log": self.log,
            "reconstruction": self.reconstruction,
            "sources": [{"stem": s.stem, "status": s.status, **{k: v for k, v in s.meta.items() if k != "tags"}}
                        for s in self.sources.values()],
        }


# ----------------------------------------------------------------------------- evidence


class Evidence:
    """Tag probabilities for one signal, restricted to the windows where it is active."""

    def __init__(self, audio: np.ndarray, mix: np.ndarray, sr: int, use_tagger: bool):
        self.activity = source_activity(audio, mix, sr)
        self.probs: np.ndarray | None = None
        if use_tagger and self.activity.active_seconds > 0:
            tags = tagger.tag_windows(audio, sr)
            keep = self._active_windows(tags["starts"])
            self.probs = tags["probs"][keep] if keep.any() else tags["probs"][:0]

    def _active_windows(self, starts: np.ndarray) -> np.ndarray:
        keep = np.zeros(len(starts), dtype=bool)
        for i, s in enumerate(starts):
            e = s + tagger.CLIP_SECONDS
            keep[i] = any(a < e and b > s for a, b in self.activity.segments)
        return keep

    def score(self, ids: tuple[int, ...]) -> float:
        if self.probs is None or not len(self.probs) or not ids:
            return 0.0
        return float(self.probs[:, list(ids)].max())

    def instrument_scores(self, allowed: set[str] | None = None) -> dict[str, float]:
        return {k: round(self.score(v), 4) for k, v in AUDIOSET.items() if allowed is None or k in allowed}

    def profile(self) -> np.ndarray:
        """Mean instrument-class profile, for similarity between signals."""
        if self.probs is None or not len(self.probs):
            return np.zeros(len(AUDIOSET))
        return np.array([float(self.probs[:, list(v)].max(axis=1).mean()) for v in AUDIOSET.values()])

    def energy_ok(self, th: Thresholds) -> bool:
        a = self.activity
        return bool(a.active_seconds >= th.min_active_seconds and a.median_active_rel_db is not None
                    and a.median_active_rel_db > th.min_rel_db)

    def energy_strong(self, th: Thresholds) -> bool:
        a = self.activity
        return bool(a.active_seconds >= th.strong_active_seconds and a.median_active_rel_db is not None
                    and a.median_active_rel_db > th.strong_rel_db)


def family_ids(family: str) -> set[str]:
    out = {k for k in AUDIOSET if (inst := get_instrument(k)) is not None and inst.family == family}
    return out


def family_score(ev: Evidence, family: str) -> float:
    return max((ev.score(AUDIOSET[k]) for k in family_ids(family)), default=0.0)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _energy(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2))


def _rel_db(x: np.ndarray, ref: np.ndarray) -> float:
    return round(10 * np.log10(max(_energy(x), 1e-20) / max(_energy(ref), 1e-20)), 2)


# ----------------------------------------------------------------------------- extractors


def ratio_mask_split(parent: np.ndarray, guides: dict[str, np.ndarray], sr: int, n_fft: int = 4096,
                     hop: int = 1024) -> dict[str, np.ndarray]:
    """Split ``parent`` with soft ratio masks built from ``guides`` (per-source estimates).

    The masks sum to one in every time-frequency bin, so the outputs sum to ``parent`` up to
    STFT round-off: nothing is invented and nothing is lost. A guide that disagrees with the
    parent only moves energy between outputs; it cannot add energy.
    """
    import librosa

    names = list(guides)
    n = parent.shape[-1]
    out = {k: np.zeros_like(parent, dtype=np.float32) for k in names}
    for c in range(parent.shape[0]):
        p = librosa.stft(parent[c], n_fft=n_fft, hop_length=hop)
        mags = [np.abs(librosa.stft(guides[k][c], n_fft=n_fft, hop_length=hop)) ** 2 for k in names]
        total = np.sum(mags, axis=0) + 1e-10
        for k, m in zip(names, mags, strict=True):
            out[k][c] = librosa.istft(p * (m / total), hop_length=hop, length=n)
    # Put STFT round-off into the last output so the sum is exact.
    err = parent - sum(out.values())
    out[names[-1]] = (out[names[-1]] + err).astype(np.float32)
    return out


def hpss_split(parent: np.ndarray, sr: int, margin: float = 2.0) -> dict[str, np.ndarray]:
    """Percussive/harmonic soft-mask split (Fitzgerald 2010 with a margin, Driedger 2014)."""
    import librosa

    n_fft, hop = 2048, 512
    n = parent.shape[-1]
    perc = np.zeros_like(parent, dtype=np.float32)
    for c in range(parent.shape[0]):
        spec = librosa.stft(parent[c], n_fft=n_fft, hop_length=hop)
        _, mask_p = librosa.decompose.hpss(np.abs(spec), margin=margin, mask=True)
        perc[c] = librosa.istft(spec * mask_p, hop_length=hop, length=n)
    return {"percussion": perc, "rest": (parent - perc).astype(np.float32)}


# ----------------------------------------------------------------------------- identity


def identify(ev: Evidence, th: Thresholds, exclude_families: set[str]) -> dict[str, Any]:
    """Name a signal from its tags. Ambiguity is reported, never resolved by guessing."""
    scores = {k: v for k, v in ev.instrument_scores().items()
              if (inst := get_instrument(k)) is not None and inst.family not in exclude_families}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    candidates = [{"id": k, "name": INSTRUMENTS[k].name, "score": v} for k, v in ranked[:5] if v > 0]
    supported = [(k, v) for k, v in ranked if v >= th.identity_tag]
    if not supported:
        return {"kind": "unresolved", "candidates": candidates}
    top, top_score = supported[0]
    runner = supported[1][1] if len(supported) > 1 else 0.0
    if len(supported) == 1 or top_score >= th.identity_margin * runner:
        return {"kind": "single", "id": top, "score": top_score, "candidates": candidates}
    families: list[str] = []
    for k, _ in supported:
        fam = INSTRUMENTS[k].family
        if fam not in families:
            families.append(fam)
    return {"kind": "group", "ids": [k for k, _ in supported[:4]], "families": families,
            "score": top_score, "candidates": candidates}


def vocal_score(ev: Evidence, th: Thresholds) -> float:
    """Vocal-family evidence with speech discounted: speech counts only above ``speech_tag``."""
    sung = max(ev.score(AUDIOSET[k]) for k in ("lead_vocal", "choir", "rap_vocal"))
    speech = ev.score(AUDIOSET["spoken_voice"])
    return max(sung, th.family_tag if speech >= th.speech_tag else 0.0, 0.0)


def identify_member(ev: Evidence, th: Thresholds, stem: str) -> dict[str, Any]:
    """Identity of a broad stem whose family is already established, chosen within that family.

    A bass stem played on a synthesizer is "Synth bass", not "Keys": the stem's role decides the
    family, the tags only choose the member. Other families in the tags are reported as contents.
    """
    family = STEM_FAMILY[stem]
    ident = identify(ev, th, exclude_families=set())
    members = {k: v for k, v in ev.instrument_scores(family_ids(family)).items() if v >= th.identity_tag}
    if "spoken_voice" in members and members["spoken_voice"] < th.speech_tag:
        del members["spoken_voice"]  # weak speech tags are formant-like timbre, not evidence of speech
    if stem == "bass" and not members and ev.score(AUDIOSET["synthesizer"]) >= th.identity_tag:
        return {"kind": "single", "id": "synth_bass", "score": round(ev.score(AUDIOSET["synthesizer"]), 4),
                "candidates": ident["candidates"]}
    if not members:
        return {"kind": "unresolved", "candidates": ident["candidates"]}
    top = max(members, key=lambda k: members[k])
    return {"kind": "single", "id": top, "score": members[top], "candidates": ident["candidates"]}


FAMILY_LABEL = {
    "vocals": "Vocals", "drums": "Drums", "percussion": "Percussion", "bass": "Bass", "guitar": "Guitars",
    "keyboards": "Keys", "strings": "Strings", "brass": "Brass", "woodwinds": "Woodwinds",
    "plucked": "Plucked strings", "other": "Other instruments", "ensemble": "Ensemble",
}


def display_for(stem: str, ident: dict[str, Any]) -> str:
    if ident["kind"] == "single":
        return INSTRUMENTS[ident["id"]].name
    if ident["kind"] == "group":
        names = [FAMILY_LABEL.get(f, f.title()) for f in ident["families"][:3]]
        return " + ".join(names)
    return "Unresolved source"


# ----------------------------------------------------------------------------- driver


SeparatorFn = Callable[[str, np.ndarray, Callable[[float], None]], dict[str, np.ndarray]]


def discover(mix: np.ndarray, sr: int, base: dict[str, np.ndarray], base_model: str,
             separate: SeparatorFn, progress: Callable[[float, str], None],
             opts: dict[str, Any] | None = None, th: Thresholds | None = None) -> Discovery:
    """Turn a broad separation into the set of sources the evidence supports.

    ``base`` is the HTDemucs output (vocals/drums/bass/other). ``separate(model, audio, cb)``
    runs another separator on demand; it is only called when tags justify it.
    """
    opts = opts or {}
    th = th or Thresholds()
    use_tags = tagger.available()
    log_: list[dict[str, Any]] = []
    sources: dict[str, Source] = {}
    rejected: dict[str, tuple[np.ndarray, Evidence]] = {}

    progress(0.0, "Listening for instruments in the full mix")
    mix_ev = Evidence(mix, mix, sr, use_tags)
    mix_scores = mix_ev.instrument_scores()
    log_.append({"step": "tag_mix", "tagger": tagger.MODEL_NAME if use_tags else "unavailable",
                 "top": _top(mix_scores)})

    # ---- 1. broad families: accept or reject each HTDemucs stem on evidence
    evid: dict[str, Evidence] = {}
    for i, name in enumerate(["vocals", "drums", "bass"]):
        if name not in base:
            continue
        progress(0.05 + 0.1 * i, f"Checking whether {name} are present")
        ev = Evidence(base[name], mix, sr, use_tags)
        evid[name] = ev
        fam = STEM_FAMILY[name]
        fscore = family_score(ev, fam) if use_tags else None
        if use_tags and name == "vocals":
            fscore = vocal_score(ev, th)
        if use_tags:
            # Vocals are the classic phantom stem (instrument bleed), so they need their own tag
            # evidence. Drums and bass may pass on strong energy alone: AudioSet's bass class is
            # weak on isolated bass, but a loud, sustained low stem is not bleed.
            ok = ev.energy_ok(th) and (fscore >= th.family_tag or  # type: ignore[operator]
                                       (name != "vocals" and ev.energy_strong(th)))
        else:
            ok = ev.energy_strong(th)
        entry = {"step": "family_check", "stem": name, "accepted": ok, "family_score": fscore,
                 "activity": ev.activity.to_dict() | {"segments": None}}
        log_.append(entry)
        if ok:
            ident = identify_member(ev, th, name) if use_tags else _no_identity()
            sources[name] = Source(name, base[name], "accepted", _meta_base(name, ev, fscore, ident, base_model, use_tags))
        else:
            rejected[name] = (base[name], ev)

    # ---- 2. the residual container: iterate targeted extraction while evidence justifies it
    residual = base["other"].astype(np.float32)
    residual_origin = f"{base_model}:other"
    tried: set[str] = set()
    for it in range(th.max_iterations):
        progress(0.35 + 0.1 * it, "Analysing what the separator left over")
        ev = Evidence(residual, mix, sr, use_tags)
        if _rel_db(residual, mix) < th.residual_floor_db or not ev.energy_ok(th):
            log_.append({"step": "residual_stop", "iteration": it, "reason": "residual below floor",
                         "residual_rel_db": _rel_db(residual, mix)})
            break
        action = _next_action(ev, mix_ev, th, tried, use_tags, opts)
        if action is None:
            log_.append({"step": "residual_stop", "iteration": it, "reason": "no extractor justified by evidence",
                         "top": _top(ev.instrument_scores())})
            break
        tried.add(action)
        if action == "htdemucs_6s":
            progress(0.4 + 0.1 * it, "Guitar or piano detected: running targeted separation")
            residual = _extract_guitar_piano(mix, residual, sr, separate, th, use_tags, sources, log_, opts)
        elif action == "hpss_percussion":
            progress(0.4 + 0.1 * it, "Auxiliary percussion detected: extracting it")
            residual = _extract_percussion(mix, residual, sr, th, use_tags, sources, log_)

    # ---- 3. name what remains
    ev = Evidence(residual, mix, sr, use_tags)
    if ev.energy_ok(th):
        exclude = {"vocals", "drums"} if "drums" in sources else {"vocals"}
        ident = identify(ev, th, exclude) if use_tags else _no_identity()
        # "Unresolved" means the tags were heard and could not name it. Without a tagger there is no
        # such evidence either way, so the source is accepted as unidentified, not demoted.
        status = "unresolved" if use_tags and ident["kind"] == "unresolved" else "accepted"
        meta = _meta_base("other", ev, None, ident, residual_origin, use_tags)
        meta["isolation"] = {"single": "residual", "group": "group", "unresolved": "unresolved"}[ident["kind"]]
        sources["other"] = Source("other", residual, status, meta)
        log_.append({"step": "residual_named", "kind": ident["kind"], "display": meta["display"],
                     "candidates": ident.get("candidates", [])[:3]})
    else:
        rejected["other"] = (residual, ev)

    # ---- 4. reassign rejected audio (bleed) so the reconstruction stays complete
    internal = None
    for name, (audio, rev) in rejected.items():
        target = _best_match(rev, sources, evid, th, use_tags)
        level = _rel_db(audio, mix)
        if target is not None and rev.activity.active_seconds > 0:
            src = sources[target]
            src.audio = (src.audio + audio).astype(np.float32)
            src.meta.setdefault("absorbed", []).append({"from": name, "rel_db": level})
            log_.append({"step": "reassign", "from": name, "to": target, "rel_db": level})
        else:
            internal = audio if internal is None else internal + audio
            log_.append({"step": "internal_residual", "from": name, "rel_db": level,
                         "reason": "no source matches its content" if rev.activity.active_seconds else "inaudible"})

    recon = _reconstruction(mix, sources, internal)
    log_.append({"step": "reconstruction", **recon})
    return Discovery(sources, internal, log_, recon, tagger.MODEL_NAME if use_tags else "unavailable")


def _no_identity() -> dict[str, Any]:
    return {"kind": "unresolved", "candidates": []}


def _top(scores: dict[str, float], n: int = 6) -> list[list[Any]]:
    return [[k, v] for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:n] if v > 0]


def _next_action(ev: Evidence, mix_ev: Evidence, th: Thresholds, tried: set[str], use_tags: bool,
                 opts: dict[str, Any]) -> str | None:
    if "htdemucs_6s" not in tried:
        if opts.get("six_stem"):
            return "htdemucs_6s"
        if use_tags:
            gp = max(family_score(ev, "guitar"), ev.score(AUDIOSET["piano"]), ev.score(AUDIOSET["electric_piano"]))
            if gp >= th.trigger_tag:
                return "htdemucs_6s"
    if "hpss_percussion" not in tried and use_tags:
        perc = max(max(ev.score(AUDIOSET[k]) for k in PERCUSSION_IDS), ev.score(GENERIC_PERCUSSION))
        if perc >= th.trigger_tag:
            return "hpss_percussion"
    return None


def _extract_guitar_piano(mix: np.ndarray, residual: np.ndarray, sr: int, separate: SeparatorFn,
                          th: Thresholds, use_tags: bool, sources: dict[str, Source],
                          log_: list[dict[str, Any]], opts: dict[str, Any]) -> np.ndarray:
    try:
        out6 = separate("htdemucs_6s", mix, lambda f: None)
    except Exception as exc:  # noqa: BLE001 - a missing optional model must not fail the job
        log_.append({"step": "targeted", "model": "htdemucs_6s", "error": str(exc)})
        return residual
    split = ratio_mask_split(residual, {"guitar": out6["guitar"], "piano": out6["piano"], "rest": out6["other"]}, sr)
    rest = split["rest"]
    for name, family in (("guitar", "guitar"), ("piano", "keyboards")):
        target = split[name]
        ev = Evidence(target, mix, sr, use_tags)
        own = family_score(ev, family) if use_tags else 0.0
        if family == "keyboards" and use_tags:
            own = max(ev.score(AUDIOSET["piano"]), ev.score(AUDIOSET["electric_piano"]),
                      ev.score(AUDIOSET["organ"]), ev.score(AUDIOSET["synthesizer"]))
        forced = bool(opts.get("six_stem")) and not use_tags
        ok = ev.energy_ok(th) and (own >= th.accept_target_tag or (forced and ev.energy_strong(th)))
        log_.append({"step": "targeted", "model": "htdemucs_6s", "target": name, "accepted": ok,
                     "target_tag": round(own, 4), "target_rel_db": _rel_db(target, mix),
                     "activity_seconds": ev.activity.active_seconds})
        if ok:
            ident: dict[str, Any] = (identify(ev, th, exclude_families={"vocals", "drums", "bass"})
                                     if use_tags else _no_identity())
            if ident["kind"] != "single" or INSTRUMENTS[ident["id"]].family != family:
                fallback = "guitar" if family == "guitar" else "piano"
                ident = {"kind": "single", "id": fallback, "score": round(own, 4),
                         "candidates": ident.get("candidates", [])}
            meta = _meta_base(name, ev, own, ident, "htdemucs_6s ratio mask of the 4-stem residual", use_tags)
            meta["isolation"] = "targeted"
            sources[name] = Source(name, target, "accepted", meta)
        else:
            rest = (rest + target).astype(np.float32)
    return rest


def _extract_percussion(mix: np.ndarray, residual: np.ndarray, sr: int, th: Thresholds, use_tags: bool,
                        sources: dict[str, Source], log_: list[dict[str, Any]]) -> np.ndarray:
    split = hpss_split(residual, sr)
    target, rest = split["percussion"], split["rest"]
    ev = Evidence(target, mix, sr, use_tags)
    scores = {k: ev.score(AUDIOSET[k]) for k in PERCUSSION_IDS}
    best = max(scores, key=lambda k: scores[k])
    own = max(scores[best], ev.score(GENERIC_PERCUSSION))
    rest_ev = Evidence(rest, mix, sr, use_tags)
    rest_perc = max(rest_ev.score(AUDIOSET[k]) for k in PERCUSSION_IDS)
    # Keep it only if the percussive part is itself tagged as percussion and taking it out
    # actually removed percussion evidence from the residual.
    ok = ev.energy_ok(th) and own >= th.accept_target_tag and rest_perc < scores[best]
    log_.append({"step": "targeted", "model": "hpss", "target": "percussion", "accepted": ok,
                 "target_tag": round(own, 4), "best": best, "residual_after": round(rest_perc, 4),
                 "target_rel_db": _rel_db(target, mix)})
    if not ok:
        return residual
    ident = {"kind": "single", "id": best if scores[best] >= th.identity_tag else "aux_percussion",
             "score": round(own, 4), "candidates": [{"id": k, "name": INSTRUMENTS[k].name, "score": round(v, 4)}
                                                    for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:3]]}
    meta = _meta_base("percussion", ev, own, ident, "HPSS percussive mask of the residual", use_tags)
    meta["isolation"] = "targeted"
    meta["limitations"].append("Harmonic/percussive masking also moves note attacks of pitched instruments "
                               "into this track; it is not a trained percussion separator.")
    sources["percussion"] = Source("percussion", target, "accepted", meta)
    return rest


def _best_match(rev: Evidence, sources: dict[str, Source], evid: dict[str, Evidence], th: Thresholds,
                use_tags: bool) -> str | None:
    if not use_tags or not sources:
        return None
    prof = rev.profile()
    best, best_sim = None, 0.0
    for name, src in sources.items():
        if src.status != "accepted":
            continue
        other = src.meta.get("_profile")
        if other is None:
            continue
        sim = _cos(prof, np.asarray(other))
        if sim > best_sim:
            best, best_sim = name, sim
    return best if best_sim >= th.reassign_similarity else None


def _meta_base(stem: str, ev: Evidence, family_score_: float | None, ident: dict[str, Any], method: str,
               use_tags: bool) -> dict[str, Any]:
    inst = get_instrument(ident["id"]) if ident.get("kind") == "single" else None
    family = inst.family if inst else (STEM_FAMILY.get(stem) or (ident.get("families") or ["other"])[0])
    limitations = []
    if inst is not None:
        limitations.append(inst.limitation)
    if not use_tags:
        limitations.append("Instrument tagger not installed: presence was decided from energy only and the "
                           "instrument is not identified. Run `stemforge-models download panns_cnn14_16k`.")
    contains = [c for c in ident.get("candidates", []) if c["score"] >= 0.12][:4]
    confidence = round(float(ident.get("score") or family_score_ or 0.0), 3) if use_tags else None
    if use_tags:
        display = display_for(stem, ident)
        if ident["kind"] == "unresolved" and stem in STEM_FAMILY:
            display = FAMILY_LABEL[STEM_FAMILY[stem]]  # family confirmed, member not named
    else:
        display = FAMILY_LABEL[STEM_FAMILY[stem]] if stem in STEM_FAMILY else "Unidentified source"
    return {
        "identity": ident.get("id"), "family": family, "display": display,
        "kind": ident["kind"], "confidence": confidence, "family_score": family_score_,
        "candidates": ident.get("candidates", []), "contains": contains,
        "activity": ev.activity.to_dict(), "separation": method,
        "isolation": "broad" if stem in STEM_FAMILY else "residual",
        "limitations": limitations, "_profile": ev.profile().round(4).tolist(),
    }


def _reconstruction(mix: np.ndarray, sources: dict[str, Source], internal: np.ndarray | None) -> dict[str, Any]:
    shown = sum((s.audio for s in sources.values()), np.zeros_like(mix))
    total = shown + (internal if internal is not None else 0.0)
    mix_e = max(_energy(mix), 1e-20)
    return {
        "residual_db": round(10 * np.log10(max(_energy(mix - total), 1e-20) / mix_e), 2),
        "unexplained_by_tracks_db": round(10 * np.log10(max(_energy(mix - shown), 1e-20) / mix_e), 2),
        "internal_residual_db": _rel_db(internal, mix) if internal is not None else None,
        "tracks": len(sources),
    }
