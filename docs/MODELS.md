# Models considered for wider instrument coverage

This note is the coverage decision for the current Stemforge build. It extends [ARCHITECTURE.md](ARCHITECTURE.md). It does not add a model just because a name appears in the instrument list. The registry the app serves at `/api/instruments` is the same decision in machine-readable form.

Checked against upstream pages in October 2026. Licenses below are what those pages say. A code license is not a weight license.

## What this build actually runs

| Piece | What it does | What it does not do | License used here |
|---|---|---|---|
| HTDemucs `htdemucs` | Stems: drums, bass, other, vocals. | No guitar, piano, or any orchestral split. | Code MIT ([adefossez/demucs](https://github.com/adefossez/demucs)). Weights: the maintainer said they are not covered by MIT and are provided for scientific purposes ([facebookresearch/demucs#327](https://github.com/facebookresearch/demucs/issues/327)). |
| HTDemucs `htdemucs_6s` | Adds guitar and piano. Upstream calls guitar okay and piano full of bleed. The 6-stem model was not trained on MUSDB; the author said the data is Meta's and declined a commercial answer ([demucs#418](https://github.com/facebookresearch/demucs/issues/418)). | Does not split acoustic/electric guitar, lead/rhythm, or keyboards other than piano. | Same code license. Weight license not stated. Optional at upload. |
| Basic Pitch 0.4 (ONNX) | Polyphonic, instrument-agnostic note transcription with pitch bends. The library already slices audio into windows of `AUDIO_N_SAMPLES` with 30 frames of overlap and stitches the activations. | Not an instrument classifier. One mix in, one set of notes out. Works best on one instrument at a time, per the project README. | Apache-2.0. Model files ship in the package. No separate weight license is stated. |
| librosa pYIN | Monophonic f0, voiced probability, and pitch bend from the contour. | Not polyphonic. | ISC. No learned weights. |
| Drum NMF in this repo | Three spectral templates (kick, snare, hi-hat), after Dittmar & Gärtner, DAFx 2014. Implemented in NumPy. | Not a neural drum transcriber. | MIT (this repo). Fixed templates, no checkpoint. |

Demucs chunk boundaries are crossfaded in `overlap_add`. A chunk that touches the start or end of the file is not faded on that edge, so samples are not shifted. Basic Pitch's own window stitch is left to that library.

## Finer drum labels (no new model)

ADTOF labels kick, snare, toms, hi-hat and cymbals, including ride. The official repository is CC BY-NC-SA 4.0. The PyTorch port does not have a LICENSE file (see the open licensing issues on [xavriley/ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch)). Those weights are not downloaded.

Instead, a hit from the three-class NMF is relabeled only when its spectrum disagrees with the template:

| Heard as | Relabel | General MIDI |
|---|---|---|
| Hi-hat, short and bright | stays closed hat | 42 |
| Hi-hat that rings | open hat | 46 |
| Hi-hat that is a short mid-frequency chick | pedal hat | 44 |
| Hi-hat that rings and has low-mid wash | crash | 49 |
| Snare with a body tone | stays snare | 38 |
| Snare-class click, almost no body | side-stick, or clap if the click is noisy | 37 or 39 |
| Kick whose spectral peak is a tom, not the sub | low / mid / high tom | 45 / 47 / 50 |

Ride, tambourine and shaker are not detected. Tambourine (GM 54) and shaker (GM 70, the maracas slot) can be drawn on the drum roll. That is manual entry, not inference.

Optional kit pieces (kick, snare, hats, residual) are Wiener masks of the drum stem from the same NMF. They match the stem's length and channel count. They are for listening. Hits are not transcribed a second time.

## Instrument families with no separator

HTDemucs has four sources, or six with `htdemucs_6s`. The following are not sources of either model, so the registry marks separation unavailable. A generic Basic Pitch or pYIN pass on the Other stem (or the bass stem, for low strings that separation already called bass) is the fallback. It does not name the instrument.

- Backing vocals, doubles, and choir stay inside the vocal stem. pYIN returns one melody. It does not separate singers.
- Electric, upright, and synth bass share the bass stem. Slap and fingerstyle are not classified.
- Acoustic and electric guitar share the guitar stem when 6-stem separation is on. Tablature is a standard-tuning fret estimate (lowest free fret). Bends and the real fingering are not detected.
- Electric piano, organ, harpsichord, synth lead, synth pad, and mallets.
- Violin, viola, cello, ensemble strings. Orchestral double bass is not distinguished from the bass stem.
- Trumpet, trombone, horn, tuba.
- Flute, piccolo, clarinet, bass clarinet, oboe, bassoon, and the four common saxophones. They are not told apart.
- Mandolin, banjo, ukulele, harp, accordion, harmonica, bagpipes.

Pitch ranges on those rows are review bounds. Notes outside them are flagged and kept.

## Reviewed and not integrated

| Project | Task | Why it is not in the default pipeline |
|---|---|---|
| python-audio-separator / UVR (BS-RoFormer, Mel-Band RoFormer, MDX-Net, VR) | Often better vocals than HTDemucs, and some instrument checkpoints exist | Code of the separator is MIT. Most community weights have no stated license. The optional adapter remains off and does not download them. |
| Tsumugi | Per-stem transcription | MIT code. Checkpoint license not stated. SheetSage beat weights are CC BY-NC 4.0. |
| MT3 | Multi-instrument transcription | Apache-2.0 code and checkpoints. JAX/T5X does not fit this install. |
| YourMT3+ | PyTorch multi-instrument transcription | Not bundled. The checkpoint was not verified here. |
| Transkun | Piano transcription | Not bundled. Piano uses Basic Pitch on the 6-stem piano stem, with the upstream bleed warning. |
| Omnizart | Drums and other transcription | Stalled on old TensorFlow. |
| madmom | Beat and downbeat models | Model license CC BY-NC-SA. Meter in Stemforge is an accent estimate on librosa beats. It stays undecided when accents are flat or when two meters are nearly tied (the bundled demo's even backbeat is one of those ties). It does not change the project's time signature. |
| Band Transcriber | A pipeline sketch (Demucs, Basic Pitch, MT3) | No stated license. No code was copied. |

## MIDI review

Review suggestions (octave leaps, pitch-range, duplicates, splintered notes, long weak sustains, dense onsets, engine disagreement) do not delete notes. A bass octave correction runs only when the subharmonic is the stronger spectral peak. The previous pitch is stored on the note and can be restored in the editor. Quantize remains an undoable edit. The generated timing is not quantized.

A second engine runs only when that option is on, and only where it answers a different question: pYIN against Basic Pitch on bass, and Basic Pitch against pYIN on vocals. Polyphonic stems are not compared with pYIN, because a monophonic pass would mostly reject chords. The two note lists are not averaged.
