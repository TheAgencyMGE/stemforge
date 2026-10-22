"""Instrument registry: what Stemforge can actually separate and transcribe.

A row in this table is not a claim that a dedicated model exists. ``separate`` and
``transcribe`` say which operation is real, and ``limitation`` is the constraint the UI
must show. Pitch ranges are review bounds, not a filter that deletes notes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Separate = Literal["isolated", "shared", "masked", "unavailable"]
Transcribe = Literal["specialized", "polyphonic", "monophonic", "events", "fallback", "unavailable"]


@dataclass(frozen=True)
class Instrument:
    id: str
    family: str
    name: str
    separate: Separate
    transcribe: Transcribe
    stem: str | None
    engines: tuple[str, ...]
    gm_program: int | None
    gm_note: int | None
    pitch_min: int | None
    pitch_max: int | None
    model: str | None
    limitation: str

    def to_public(self) -> dict[str, Any]:
        data = asdict(self)
        data["engines"] = list(self.engines)
        data["summary"] = _summary(self)
        return data


def _summary(i: Instrument) -> str:
    sep = {
        "isolated": f"Separated as the “{i.stem}” stem by {i.model}.",
        "shared": f"Not separated from the other sounds in the “{i.stem}” stem.",
        "masked": "Optional spectral mask of the drum stem. Not a learned instrument model.",
        "unavailable": "No separation model in this build produces this instrument on its own.",
    }[i.separate]
    tr = {
        "specialized": f"Transcribed with {', '.join(i.engines)} using instrument-specific settings.",
        "polyphonic": f"Transcribed with polyphonic {', '.join(i.engines)}.",
        "monophonic": f"Transcribed as one melody with {', '.join(i.engines)}.",
        "events": "Drum hits are classified on the drum stem and written as General MIDI percussion.",
        "fallback": "Only the generic polyphonic transcriber can run, on whichever stem contains the sound.",
        "unavailable": "No transcription path is offered.",
    }[i.transcribe]
    return f"{sep} {tr} {i.limitation}"


def _inst(
    id: str, family: str, name: str, separate: Separate, transcribe: Transcribe,
    limitation: str, *, stem: str | None = None, engines: tuple[str, ...] = (),
    gm_program: int | None = None, gm_note: int | None = None,
    pitch_min: int | None = None, pitch_max: int | None = None, model: str | None = None,
) -> Instrument:
    return Instrument(
        id=id, family=family, name=name, separate=separate, transcribe=transcribe, stem=stem,
        engines=engines, gm_program=gm_program, gm_note=gm_note, pitch_min=pitch_min,
        pitch_max=pitch_max, model=model, limitation=limitation,
    )


# Isolated stems are exactly the HTDemucs source names. Everything else is shared, masked, or absent.
_ISOLATED_STEMS = {"vocals", "drums", "bass", "other", "guitar", "piano"}

INSTRUMENTS: dict[str, Instrument] = {
    i.id: i for i in [
        _inst("lead_vocal", "vocals", "Lead vocal", "isolated", "monophonic",
              "One melody is taken from the vocal stem. A mixed vocal stem is not a set of isolated singers.",
              stem="vocals", engines=("pyin", "basic-pitch"), gm_program=53, pitch_min=45, pitch_max=86,
              model="htdemucs"),
        _inst("backing_vocal", "vocals", "Backing vocal", "shared", "fallback",
              "Backing vocals stay inside the vocal stem. A second polyphonic pass can show extra pitches; it does not separate singers.",
              stem="vocals", engines=("basic-pitch",), gm_program=53, pitch_min=45, pitch_max=86,
              model=None),
        _inst("choir", "vocals", "Choir / ensemble vocal", "shared", "fallback",
              "Choir is not a Demucs source. It remains inside vocals or other, and is not split into parts.",
              stem="vocals", engines=("basic-pitch",), gm_program=52, pitch_min=36, pitch_max=84,
              model=None),
        _inst("drum_kit", "drums", "Drum kit", "isolated", "events",
              "The stem is the whole kit. Kick, snare and hi-hat come from semi-adaptive NMF templates. Finer classes are spectral labels on those hits, not a neural drum model.",
              stem="drums", engines=("drum-nmf",), gm_note=36, model="htdemucs"),
        _inst("kick", "drums", "Kick", "masked", "events",
              "GM 36. The optional kit mask is an NMF Wiener mask of the drum stem, sample-aligned, with bleed.",
              stem="kit_kick", engines=("drum-nmf",), gm_note=36, model="drum-nmf"),
        _inst("snare", "drums", "Snare", "masked", "events",
              "GM 38. Side-stick (GM 37) is used only when a snare-class hit has a click and almost no body tone.",
              stem="kit_snare", engines=("drum-nmf",), gm_note=38, model="drum-nmf"),
        _inst("sidestick", "drums", "Side-stick", "unavailable", "events",
              "Not its own template. A snare-class hit can be relabeled GM 37 when the spectrum is a short click without a snare body.",
              stem="drums", engines=("drum-nmf",), gm_note=37, model="drum-nmf"),
        _inst("closed_hat", "drums", "Closed hi-hat", "masked", "events",
              "GM 42, the default hi-hat class. The hat mask also contains open hats and cymbal bleed.",
              stem="kit_hats", engines=("drum-nmf",), gm_note=42, model="drum-nmf"),
        _inst("pedal_hat", "drums", "Pedal hi-hat", "unavailable", "events",
              "GM 44 only when a hi-hat-class hit is a short mid-frequency chick rather than a bright closed hat.",
              stem="drums", engines=("drum-nmf",), gm_note=44, model="drum-nmf"),
        _inst("open_hat", "drums", "Open hi-hat", "unavailable", "events",
              "GM 46 when a hi-hat-class hit rings. A long broadband wash is labeled crash (GM 49) instead.",
              stem="drums", engines=("drum-nmf",), gm_note=46, model="drum-nmf"),
        _inst("crash", "drums", "Crash cymbal", "unavailable", "events",
              "GM 49 from hi-hat-class hits that ring and contain low-mid wash. There is no separate cymbal model.",
              stem="drums", engines=("drum-nmf",), gm_note=49, model="drum-nmf"),
        _inst("ride", "drums", "Ride cymbal", "unavailable", "events",
              "No ride-specific detector. A ringing bright cymbal is labeled open hat or crash, not ride.",
              stem="drums", engines=("drum-nmf",), gm_note=51, model=None),
        _inst("tom_high", "drums", "High tom", "unavailable", "events",
              "GM 50 when a kick- or snare-class hit is tonal above about 220 Hz, or a residual tonal onset is found there.",
              stem="drums", engines=("drum-nmf",), gm_note=50, model="drum-nmf"),
        _inst("tom_mid", "drums", "Mid tom", "unavailable", "events",
              "GM 47 for a tonal tom-like hit around 140–220 Hz.",
              stem="drums", engines=("drum-nmf",), gm_note=47, model="drum-nmf"),
        _inst("tom_low", "drums", "Low tom", "unavailable", "events",
              "GM 45 for a tonal tom-like hit around 80–140 Hz that is not a sub-heavy kick.",
              stem="drums", engines=("drum-nmf",), gm_note=45, model="drum-nmf"),
        _inst("clap", "drums", "Clap", "unavailable", "events",
              "GM 39 when a snare-class hit is a short broadband burst without snare body. Not a trained clap model.",
              stem="drums", engines=("drum-nmf",), gm_note=39, model="drum-nmf"),
        _inst("tambourine", "drums", "Tambourine", "unavailable", "unavailable",
              "No detector. Draw GM 54 on the drum roll if you want the note. The hat template may absorb the jingles.",
              stem="drums", gm_note=54, model=None),
        _inst("shaker", "drums", "Shaker", "unavailable", "unavailable",
              "No detector. The closest General MIDI slot is maracas, GM 70, which you can draw. It is not inferred.",
              stem="drums", gm_note=70, model=None),
        _inst("bass", "bass", "Bass", "isolated", "specialized",
              "One bass stem. Electric, upright and synth bass are not split. Basic Pitch is band-limited to 30–500 Hz; octave corrections are marked and reversible.",
              stem="bass", engines=("basic-pitch", "pyin"), gm_program=33, pitch_min=24, pitch_max=64,
              model="htdemucs"),
        _inst("electric_bass", "bass", "Electric bass", "shared", "specialized",
              "Uses the bass stem. Fingerstyle and slap are not classified.",
              stem="bass", engines=("basic-pitch", "pyin"), gm_program=33, pitch_min=24, pitch_max=64,
              model=None),
        _inst("upright_bass", "bass", "Upright bass", "shared", "specialized",
              "Uses the bass stem. There is no upright-bass separator.",
              stem="bass", engines=("basic-pitch", "pyin"), gm_program=32, pitch_min=28, pitch_max=60,
              model=None),
        _inst("synth_bass", "bass", "Synth bass", "shared", "specialized",
              "Uses the bass stem when separation puts it there. A bass synth that stays in the mix may land in Other.",
              stem="bass", engines=("basic-pitch", "pyin"), gm_program=38, pitch_min=24, pitch_max=72,
              model=None),
        _inst("guitar", "guitar", "Guitar", "isolated", "polyphonic",
              "Only htdemucs_6s outputs a guitar stem, and it is one stem. Clean, distorted, acoustic and lead are not split. Tablature is a standard-tuning fret estimate.",
              stem="guitar", engines=("basic-pitch", "pyin"), gm_program=27, pitch_min=36, pitch_max=88,
              model="htdemucs_6s"),
        _inst("acoustic_guitar", "guitar", "Acoustic guitar", "shared", "polyphonic",
              "Not separated from electric guitar.",
              stem="guitar", engines=("basic-pitch",), gm_program=25, pitch_min=40, pitch_max=88,
              model=None),
        _inst("electric_guitar", "guitar", "Electric guitar", "shared", "polyphonic",
              "Not separated from acoustic guitar. Distortion is not classified.",
              stem="guitar", engines=("basic-pitch",), gm_program=27, pitch_min=40, pitch_max=88,
              model=None),
        _inst("piano", "keyboards", "Acoustic piano", "isolated", "polyphonic",
              "htdemucs_6s piano. Upstream reports heavy bleed and artifacts, and the track is labeled that way.",
              stem="piano", engines=("basic-pitch", "pyin"), gm_program=0, pitch_min=21, pitch_max=108,
              model="htdemucs_6s"),
        _inst("electric_piano", "keyboards", "Electric piano", "unavailable", "fallback",
              "No electric-piano separator. It usually remains in Other and is transcribed there, without an instrument label.",
              stem="other", engines=("basic-pitch",), gm_program=4, pitch_min=28, pitch_max=96,
              model=None),
        _inst("organ", "keyboards", "Organ", "unavailable", "fallback",
              "No organ separator. Sustained organ tones in Other are generic polyphonic notes.",
              stem="other", engines=("basic-pitch",), gm_program=19, pitch_min=36, pitch_max=96,
              model=None),
        _inst("harpsichord", "keyboards", "Harpsichord", "unavailable", "fallback",
              "No harpsichord model.",
              stem="other", engines=("basic-pitch",), gm_program=6, pitch_min=29, pitch_max=89,
              model=None),
        _inst("synth_lead", "keyboards", "Synth lead", "unavailable", "fallback",
              "No synth-lead separator. A monophonic line in Other can be re-transcribed with pYIN.",
              stem="other", engines=("basic-pitch", "pyin"), gm_program=81, pitch_min=36, pitch_max=96,
              model=None),
        _inst("synth_pad", "keyboards", "Synth pad", "unavailable", "fallback",
              "No pad separator. Long pad notes are easy for a transcriber to smear; review long sustains.",
              stem="other", engines=("basic-pitch",), gm_program=88, pitch_min=36, pitch_max=96,
              model=None),
        _inst("mallet", "keyboards", "Mallet (vibraphone, marimba, xylophone)", "unavailable", "fallback",
              "No mallet separator or mallet-specific transcriber.",
              stem="other", engines=("basic-pitch",), gm_program=11, pitch_min=41, pitch_max=96,
              model=None),
        _inst("violin", "strings", "Violin", "unavailable", "fallback",
              "No string separator. A solo line in Other can use pYIN; an ensemble stays polyphonic Basic Pitch. Vibrato is pitch bend only on monophonic passes.",
              stem="other", engines=("pyin", "basic-pitch"), gm_program=40, pitch_min=55, pitch_max=96,
              model=None),
        _inst("viola", "strings", "Viola", "unavailable", "fallback",
              "No viola model.", stem="other", engines=("pyin", "basic-pitch"), gm_program=41,
              pitch_min=48, pitch_max=88, model=None),
        _inst("cello", "strings", "Cello", "unavailable", "fallback",
              "No cello model. Low strings may be pulled into the bass stem.",
              stem="other", engines=("pyin", "basic-pitch"), gm_program=42, pitch_min=36, pitch_max=76,
              model=None),
        _inst("double_bass_orch", "strings", "Orchestral double bass", "shared", "specialized",
              "If separation treats it as bass, it uses the bass stem. It is not distinguished from electric bass.",
              stem="bass", engines=("basic-pitch", "pyin"), gm_program=43, pitch_min=24, pitch_max=60,
              model=None),
        _inst("string_ensemble", "strings", "String ensemble", "unavailable", "fallback",
              "No ensemble separator. Polyphonic notes in Other are not assigned to violin, viola or cello.",
              stem="other", engines=("basic-pitch",), gm_program=48, pitch_min=36, pitch_max=96,
              model=None),
        _inst("trumpet", "brass", "Trumpet", "unavailable", "fallback",
              "No brass separator. A solo line can use pYIN on Other.",
              stem="other", engines=("pyin", "basic-pitch"), gm_program=56, pitch_min=52, pitch_max=82,
              model=None),
        _inst("trombone", "brass", "Trombone", "unavailable", "fallback",
              "No trombone model.", stem="other", engines=("pyin", "basic-pitch"), gm_program=57,
              pitch_min=40, pitch_max=72, model=None),
        _inst("french_horn", "brass", "French horn", "unavailable", "fallback",
              "No horn model.", stem="other", engines=("pyin", "basic-pitch"), gm_program=60,
              pitch_min=41, pitch_max=77, model=None),
        _inst("tuba", "brass", "Tuba", "unavailable", "fallback",
              "No tuba model. Low brass may fall into the bass stem and be transcribed as bass.",
              stem="other", engines=("basic-pitch", "pyin"), gm_program=58, pitch_min=28, pitch_max=60,
              model=None),
        _inst("flute", "woodwinds", "Flute", "unavailable", "fallback",
              "No woodwind separator.", stem="other", engines=("pyin", "basic-pitch"), gm_program=73,
              pitch_min=60, pitch_max=96, model=None),
        _inst("piccolo", "woodwinds", "Piccolo", "unavailable", "fallback",
              "No piccolo model.", stem="other", engines=("pyin",), gm_program=72,
              pitch_min=74, pitch_max=108, model=None),
        _inst("clarinet", "woodwinds", "Clarinet", "unavailable", "fallback",
              "No clarinet model.", stem="other", engines=("pyin", "basic-pitch"), gm_program=71,
              pitch_min=50, pitch_max=89, model=None),
        _inst("bass_clarinet", "woodwinds", "Bass clarinet", "unavailable", "fallback",
              "No bass-clarinet model.", stem="other", engines=("pyin", "basic-pitch"), gm_program=71,
              pitch_min=37, pitch_max=72, model=None),
        _inst("oboe", "woodwinds", "Oboe", "unavailable", "fallback",
              "No oboe model.", stem="other", engines=("pyin",), gm_program=68,
              pitch_min=58, pitch_max=91, model=None),
        _inst("bassoon", "woodwinds", "Bassoon", "unavailable", "fallback",
              "No bassoon model.", stem="other", engines=("pyin", "basic-pitch"), gm_program=70,
              pitch_min=34, pitch_max=72, model=None),
        _inst("sax_alto", "woodwinds", "Alto saxophone", "unavailable", "fallback",
              "No saxophone model, and alto/tenor/soprano/baritone are not distinguished.",
              stem="other", engines=("pyin", "basic-pitch"), gm_program=65, pitch_min=49, pitch_max=80,
              model=None),
        _inst("sax_tenor", "woodwinds", "Tenor saxophone", "unavailable", "fallback",
              "No tenor-sax model.", stem="other", engines=("pyin", "basic-pitch"), gm_program=66,
              pitch_min=44, pitch_max=75, model=None),
        _inst("sax_soprano", "woodwinds", "Soprano saxophone", "unavailable", "fallback",
              "No soprano-sax model.", stem="other", engines=("pyin",), gm_program=64,
              pitch_min=56, pitch_max=88, model=None),
        _inst("sax_baritone", "woodwinds", "Baritone saxophone", "unavailable", "fallback",
              "No baritone-sax model.", stem="other", engines=("pyin", "basic-pitch"), gm_program=67,
              pitch_min=36, pitch_max=68, model=None),
        _inst("recorder", "woodwinds", "Recorder", "unavailable", "fallback",
              "No recorder model.", stem="other", engines=("pyin",), gm_program=74,
              pitch_min=60, pitch_max=96, model=None),
        _inst("mandolin", "plucked", "Mandolin", "unavailable", "fallback",
              "No mandolin model.", stem="other", engines=("basic-pitch",), gm_program=25,
              pitch_min=55, pitch_max=88, model=None),
        _inst("banjo", "plucked", "Banjo", "unavailable", "fallback",
              "No banjo model.", stem="other", engines=("basic-pitch",), gm_program=105,
              pitch_min=50, pitch_max=84, model=None),
        _inst("ukulele", "plucked", "Ukulele", "unavailable", "fallback",
              "No ukulele model.", stem="other", engines=("basic-pitch",), gm_program=24,
              pitch_min=55, pitch_max=88, model=None),
        _inst("harp", "plucked", "Harp", "unavailable", "fallback",
              "No harp model.", stem="other", engines=("basic-pitch",), gm_program=46,
              pitch_min=23, pitch_max=103, model=None),
        _inst("accordion", "other", "Accordion", "unavailable", "fallback",
              "No accordion model.", stem="other", engines=("basic-pitch",), gm_program=21,
              pitch_min=41, pitch_max=89, model=None),
        _inst("harmonica", "other", "Harmonica", "unavailable", "fallback",
              "No harmonica model.", stem="other", engines=("pyin",), gm_program=22,
              pitch_min=60, pitch_max=96, model=None),
        _inst("bagpipes", "other", "Bagpipes", "unavailable", "fallback",
              "No bagpipe model. A drone may become a long note; that is not a chanter transcription.",
              stem="other", engines=("pyin",), gm_program=109, pitch_min=50, pitch_max=84,
              model=None),
        _inst("residual", "other", "Residual accompaniment", "isolated", "polyphonic",
              "The Other stem is whatever the separator did not assign. Notes are not labeled by instrument.",
              stem="other", engines=("basic-pitch", "pyin"), gm_program=48, pitch_min=21, pitch_max=108,
              model="htdemucs"),
        _inst("full_mix", "other", "Full mix", "unavailable", "polyphonic",
              "Optional Basic Pitch pass on the original mix, for comparison. It is not a separation.",
              stem="mix", engines=("basic-pitch",), gm_program=0, pitch_min=21, pitch_max=108,
              model="basic-pitch"),
    ]
}

# Models that were reviewed and left out of the default pipeline. Do not wire these in without a new licensing decision.
EXCLUDED_MODELS: list[dict[str, str]] = [
    {
        "id": "adtof",
        "task": "Drum transcription (kick, snare, toms, hi-hat, cymbals)",
        "reason": "The official ADTOF repository is CC BY-NC-SA 4.0. ADTOF-pytorch ships weights and, as of 2026, has no LICENSE file. Noncommercial terms block the default distribution.",
    },
    {
        "id": "madmom",
        "task": "Beat, downbeat and some transcription models",
        "reason": "Code is BSD; the neural models are CC BY-NC-SA. Not used. Meter here is an accent estimate on librosa beats.",
    },
    {
        "id": "omnizart",
        "task": "Multi-instrument transcription including drums",
        "reason": "Stalled on TensorFlow 2.5-era dependencies. Not integrated.",
    },
    {
        "id": "mt3",
        "task": "Multi-instrument transcription",
        "reason": "JAX/T5X stack. Checkpoints are Apache-2.0, but the runtime does not fit this install. Not integrated.",
    },
    {
        "id": "yourmt3",
        "task": "PyTorch multi-instrument transcription",
        "reason": "Candidate only. Not bundled; checkpoint and runtime were not verified in this environment.",
    },
    {
        "id": "tsumugi",
        "task": "Per-stem transcription",
        "reason": "MIT code. Checkpoint license is not stated, and the SheetSage beat model it can use is CC BY-NC 4.0. Not bundled.",
    },
    {
        "id": "transkun",
        "task": "Piano transcription",
        "reason": "Not bundled. Piano uses Basic Pitch on the htdemucs_6s piano stem.",
    },
    {
        "id": "uvr-weights",
        "task": "BS-RoFormer, Mel-RoFormer, MDX-Net, VR vocal and instrument models",
        "reason": "python-audio-separator can load them, and the adapter remains optional. Most community weights have no stated license, so they are not downloaded or selected by default.",
    },
]

FAMILY_ORDER = [
    "vocals", "drums", "bass", "guitar", "keyboards", "strings", "brass", "woodwinds", "plucked", "other",
]

# Pitch bounds used when reviewing a stem. Notes outside the span are flagged, never deleted.
STEM_PITCH_RANGE: dict[str, tuple[int, int]] = {
    "vocals": (45, 86),
    "bass": (24, 64),
    "guitar": (36, 88),
    "piano": (21, 108),
    "other": (21, 108),
    "mix": (21, 108),
}

STEM_BLURB: dict[str, str] = {
    "vocals": "Vocal stem from HTDemucs. pYIN transcribes one melody. Mixed singers are not separated.",
    "drums": "Whole kit from HTDemucs. NMF labels kick, snare and hi-hat; open hat, pedal hat, crash, tom and clap are spectral relabels when the hit does not match the coarse template.",
    "bass": "Bass stem from HTDemucs. Not split into electric, upright or synth. Octave corrections keep the original pitch on the note.",
    "guitar": "Guitar stem from htdemucs_6s only. One stem, not a rhythm/lead split. Tablature is a standard-tuning estimate.",
    "piano": "Piano stem from htdemucs_6s. Upstream reports heavy bleed. Transcription is polyphonic Basic Pitch, not a piano-only model.",
    "other": "Residual stem. Instruments here are not identified. Basic Pitch transcribes pitches; it does not name the source.",
    "mix": "Original mix, not a stem. Basic Pitch puts every instrument it hears on one track.",
    "kit_kick": "NMF mask of the drum stem for listening. Sample-aligned. Hits stay on the Drums MIDI track.",
    "kit_snare": "NMF mask of the drum stem for listening. Sample-aligned. Hits stay on the Drums MIDI track.",
    "kit_hats": "NMF mask of the drum stem for listening. Open hats and cymbals can remain in this mask.",
    "kit_perc": "Whatever the three drum templates did not explain, as a mask. Not a classified percussion stem.",
}


def catalog() -> list[dict[str, Any]]:
    return [INSTRUMENTS[i].to_public() for i in INSTRUMENTS]


def families() -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        rows = [INSTRUMENTS[i].to_public() for i in INSTRUMENTS if INSTRUMENTS[i].family == family]
        if rows:
            grouped.append({"id": family, "instruments": rows})
    return grouped


def get_instrument(instrument_id: str) -> Instrument | None:
    return INSTRUMENTS.get(instrument_id)


def stem_blurb(stem: str) -> str:
    if stem in STEM_BLURB:
        return STEM_BLURB[stem]
    return "This stem has no dedicated instrument model."


def validate_registry() -> None:
    """Raise ValueError if the registry claims a separator or a pitch map it does not have."""
    seen: set[str] = set()
    for inst in INSTRUMENTS.values():
        if inst.id in seen:
            raise ValueError(f"duplicate instrument id {inst.id}")
        seen.add(inst.id)
        if inst.separate == "isolated":
            if inst.stem not in _ISOLATED_STEMS:
                raise ValueError(f"{inst.id} claims an isolated stem {inst.stem}")
            if inst.model not in {"htdemucs", "htdemucs_6s"}:
                raise ValueError(f"{inst.id} isolated stem is not an HTDemucs source")
        if inst.separate == "masked" and inst.model != "drum-nmf":
            raise ValueError(f"{inst.id} mask is not the drum NMF")
        if inst.transcribe == "unavailable" and inst.engines:
            raise ValueError(f"{inst.id} lists engines but cannot transcribe")
        if inst.gm_note is not None and not 0 <= inst.gm_note <= 127:
            raise ValueError(f"{inst.id} GM note out of range")
        if inst.gm_program is not None and not 0 <= inst.gm_program <= 127:
            raise ValueError(f"{inst.id} GM program out of range")
        if inst.pitch_min is not None and inst.pitch_max is not None and inst.pitch_min > inst.pitch_max:
            raise ValueError(f"{inst.id} pitch range inverted")
        if inst.family in {"strings", "brass", "woodwinds", "plucked"} and inst.separate == "isolated":
            raise ValueError(f"{inst.id} claims isolation without a model")
    for blocked in ("adtof", "madmom", "omnizart", "tsumugi"):
        if any(blocked in (i.model or "") for i in INSTRUMENTS.values()):
            raise ValueError(f"{blocked} must not be a default model")
