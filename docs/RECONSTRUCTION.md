# Rebuilding a song: MIDI for Soundtrap, stems, and what fidelity is measurable

Checked October 23, 2026. This document covers the Soundtrap workflow, the optional research
transcription engines, and the benchmark that decided how they are used. Numbers come from
`backend/scripts/evaluate_slakh.py` and `backend/scripts/summarize_slakh.py`; see
[EVALUATION.md](EVALUATION.md) for the full tables.

## Two different fidelity targets

| Target | What it means | How Stemforge measures it |
|---|---|---|
| **Symbolic fidelity** | The MIDI contains the notes that were played: pitch, onset, offset, velocity, drum hits. | Note F1 (onset; onset+offset), velocity correlation and drum onset F1 against reference MIDI. |
| **Audio fidelity** | The stems sound like the original parts. | SI-SDR of each separated stem against the true stem (benchmark only), mixture reconstruction error (every project). |
| **MIDI render fidelity** | How close the MIDI sounds to the stem when played through a General MIDI instrument. | Render through FluidSynth + MuseScore General; pitch match (chroma), rhythm match (onset envelope), overall sound difference (log-mel). |

These are separate. A perfect transcription played on a Soundtrap instrument still does not sound
like the record: MIDI carries notes, not samples, effects, mixing or mastering. The benchmark puts a
number on that: rendering the *true* MIDI and comparing it with the *true* audio gives the gap that
no transcription can close. Keep the WAV stems for the original sound.

## The Soundtrap workflow

Facts from Soundtrap's help centre (October 2026):

* MIDI is imported by dragging a `.mid` file onto the timeline.
  ([article](https://support.soundtrap.com/hc/en-us/articles/205659501-How-to-Import-and-Export-MIDI-Files-in-Soundtrap))
* Audio imports: mp3, wav, aif, mp4, m4a, ogg, aac. **FLAC is not listed.**
  ([article](https://support.soundtrap.com/hc/en-us/articles/205664281-What-Audio-File-Types-Does-Soundtrap-Support))
* No file-size limit; the practical limit is memory.
  ([article](https://support.soundtrap.com/hc/en-us/articles/360010469920-What-Is-the-File-Size-Limit-for-Importing-Files-into-Soundtrap))
* One project tempo, set in BPM (default 120). Changing it offers to adjust audio, so MIDI follows
  beats while audio follows time. No tempo-map support is documented.

What the **Soundtrap package** (Export → Rebuild in Soundtrap) contains, and why:

| File | Format | Reason |
|---|---|---|
| `NN Track.mid` | Type-1 SMF, one track, **integer tempo** | Soundtrap has one tempo, typed by the user. Notes are converted to ticks at that integer tempo, so they keep their exact time in seconds when the project is set to it. |
| `NN Track.wav` | 24-bit PCM, full song length | Lossless and importable (FLAC is not). Every stem starts at 0:00, so they line up. Checked bit-exact against the stored stem. |
| `SOUNDTRAP.md` | Steps | Set the BPM first, then import; an instrument suggestion per track (its General MIDI program and family). |
| `export_manifest.json` | Facts | Tempo used vs detected, sample rate, bit depth, frames, PCM hashes, render fidelity per track. |

Drums are on General MIDI channel 10 with GM note numbers. Pitch bends are written for monophonic
passages (MIDI bends act on a whole channel).

**What was verified and what was not.** The exported MIDI was parsed by four independent readers:
mido, symusic (C++), pretty_midi, and `@tonejs/midi`, a browser MIDI parser of the kind web DAWs use
(`frontend/tests/unit/midi-parsers.test.ts`). Track names, programs, drum channel, tempo, note times
within one tick, velocities and bends all agree. **It was not imported into Soundtrap itself**: that
needs a Soundtrap account and was not done, so how Soundtrap maps channel-10 drums to its drum
instruments is unverified.

For other DAWs, the MIDI export can instead write a **tempo map** from the detected beats (Export →
"MIDI tempo map from detected beats"), so bar lines follow a song that speeds up or slows down. Note
times are identical either way; a test checks that a 90→130 BPM performance lands on the grid and that
an independent parser recovers every note time within 3 ms.

## Research transcription engines

Optional, each in its own environment, called as a separate program, never redistributed. Install
with `scripts/setup-research.ps1`. They appear in the track inspector only when installed.

| Engine | Model | Licence | Status in this build |
|---|---|---|---|
| `transkun` | Transkun 2.0.1, piano | Code MIT; checkpoint licence not stated | Integrated and benchmarked. With `STEMFORGE_RESEARCH_ENGINES=true` it is the automatic choice for piano sources. |
| `yourmt3` | YourMT3+ YPTF.MoE+Multi (noPS) | Code GPL-3.0 (separate program); checkpoint Apache-2.0 per its card; unpickled by the authors' loader inside its own environment | Integrated (adapter, runner, attribution) and benchmarked. Selectable per track; not in the automatic policy (see below). |
| `muscriptor` | MuScriptor 0.4.0 (Kyutai, Mirelo) | Code MIT; weights CC BY-NC 4.0 + model-card conditions; gated | Adapter, instrument conditioning and velocity derivation are implemented and tested with a stub engine. **The real model was not run**: its weights require each user to accept the conditions with their own Hugging Face account. |

To enable MuScriptor yourself: sign in at huggingface.co, accept the conditions on
`MuScriptor/muscriptor-small` (and `-medium`/`-large` if wanted), then run
`%USERPROFILE%\.sfvenv-muscriptor\Scripts\hf.exe auth login`. Stemforge conditions it on the
instruments discovery found (`--instruments`) and takes velocity from the stem's loudness, because the
model emits none.

Engines without per-note probabilities (all three) produce `confidence: null` with the flag
`no_model_confidence`; a score is never invented. Raw engine output is stored per track
(`notes_raw` artifact) before any refinement.

## Measured on BabySlakh (20 songs, details in EVALUATION.md)

| Per-track policy on separated sources | Bass F1 | Keyboards F1 | Guitar F1 | All notes, onset+offset F1 |
|---|---|---|---|---|
| Default (Basic Pitch, pYIN, NMF) | 0.61 | 0.31 | 0.27 | 0.15 |
| Research engines on, YourMT3+ installed (hybrid) | 0.61 | 0.47 | 0.46 | 0.66 |

* Separation limits everything in dense mixes: Transkun reaches 0.97 on a true piano stem but 0.43 on a
  separated one.
* Even perfect MIDI rendered through a General MIDI soundfont differs from the real instrument by
  8.4 dB (log-mel) with 0.89 rhythm correlation: the floor for any instrument choice in Soundtrap.
* On the synthetic demo song the research policy is *worse* than the default (piano render pitch match
  0.72 vs 0.84). No engine wins everywhere; the policy follows the larger, more realistic benchmark, and
  every track can be switched back to another engine in the inspector.
* Choosing engines automatically by render fidelity was tested and rejected (it picked the better engine
  for 25 of 53 sources).
