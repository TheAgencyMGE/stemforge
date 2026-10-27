# Research: Suno-style stem separation, and what an open-source build can reproduce

Checked October 22, 2026. "Verified" means the licence or claim was read on the primary page
linked here during this work; anything else is marked as not verified. A code licence is not a
weight licence.

## 1. What Suno has published

Primary sources:

* Blog, "We've made improvements to Stem Separation" (June 11, 2026): https://blog.suno.com/blog/stem-separation-updates
  (the `about.suno.com` URL from the brief did not resolve from this machine; this blog is the same post)
* Release note, Jun 11, 2026: https://suno.com/release-notes/advanced-stems
* Help centre, Advanced Stem Separation: https://help.suno.com/en/articles/12702337
* Product page: https://suno.com/stem-separation
* Release notes index (Studio 2.0 on Aug 13, MIDI fixes Sep 2, MIDI-to-audio and MIDI import on Sep 17, 2026): https://suno.com/release-notes

What these pages actually say:

| Mode | Documented behaviour |
|---|---|
| **Auto Split** | The original model. Up to 12 stems in fixed categories (the pages name drums, bass, guitar, keys, woodwinds, lead vocals, "other"). Suno calls it reliable but less precise than the new modes, and says labels are not always correct. |
| **Split from Mix** | The user names one instrument or voice; the output is that stem plus a complement ("everything else"). |
| **Advanced Split** | Pick from "nearly 100" instruments (examples run from drum kit to didgeridoo). Premier plan only; some granular options are "still in development". Requesting an instrument that is not in the song can consume credits without a usable stem. |
| **How stems are made** | The blog says the system listens to the track and *regenerates* each stem with Suno's latest model, in contrast to tools that cut a finished mix into frequency slices. It claims stems free of bleed and artifacts. No architecture, paper, metric or independent test is published. |

What is **not** public: model architecture, training data, conditioning method, how the
"nearly 100" classes map to models, and any measurement of fidelity to the original recording.
Anything beyond the table above is inference. In particular, a regenerated stem is new audio that
*resembles* the part; it is not guaranteed to be sample-faithful to the recording, which is why
third-party commentary advises checking regenerated stems against the master.

## 2. The building blocks in public research

| Capability | Representative open work | What it gives | Licence (verified where marked) | Fit for Stemforge |
|---|---|---|---|---|
| Fixed multi-stem separation | Demucs / HTDemucs (`htdemucs`, `htdemucs_6s`) | 4 or 6 broad stems, strong SDR | Code MIT. Weights: no licence in the HF repos; maintainer said they are provided for scientific purposes (see THIRD_PARTY_NOTICES). | **In use**, unchanged. |
| Band-split transformers | BS-RoFormer, Mel-Band RoFormer (via ZFTurbo/Music-Source-Separation-Training, python-audio-separator) | Best public vocal/instrumental SDR; community instrument checkpoints | MSST code MIT (verified). Its pretrained-model list states **no licence for any weight** (verified: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md). | Not bundled. Optional adapter remains off. |
| Drum sub-separation | DrumSep (htdemucs / MDX23C variants listed in MSST) | kick / snare / toms / hats / cymbals | No weight licence stated (same page). | Not bundled; the existing NMF masks stay. |
| **Query-conditioned music separation** | Banquet / query-bandit (Watcharasupat & Lerch, ISMIR 2024, https://arxiv.org/abs/2406.18747) | Split-from-mix by audio query; narrow classes (acoustic guitar, reeds, organ) | Code MIT (verified: https://github.com/kwatcharasupat/query-bandit). Checkpoints **CC BY-NC-SA 4.0** (verified: https://zenodo.org/records/13694558). | Excluded from the default build (non-commercial). Architecturally the closest public analogue to "Split from Mix"/"Advanced Split". |
| Text-queried separation | AudioSep (https://github.com/Audio-AGI/AudioSep) | Separate "anything you describe", 32 kHz, general audio | Code MIT (verified). Checkpoint licence **not stated** on the repo. | Not bundled. Trained on general sound events; music-instrument quality unmeasured here. |
| Audio tagging (instrument identification) | PANNs Cnn14 (Kong et al. 2020) | 527 AudioSet classes, about 90 of them instruments and voice types | Code MIT (verified: LICENSE.MIT, Copyright Qiuqiang Kong). Weights **CC BY 4.0** (verified: https://zenodo.org/records/3987831). | **Integrated** for source discovery (attribution in THIRD_PARTY_NOTICES). |
| Text–audio embeddings | LAION-CLAP (https://github.com/LAION-AI/CLAP) | Zero-shot labels for arbitrary text | Code CC0-1.0 (verified). Checkpoint licence not stated; part of the training data is research-only. | Not bundled. |
| Multi-instrument transcription | MT3 (Apache-2.0, JAX), YourMT3/YourMT3+ (https://github.com/mimbres/YourMT3) | Multi-track MIDI from a mix | YourMT3 code **GPL-3.0** (verified); checkpoint licence and hosting not stated. | Not bundled (GPL is incompatible with this MIT distribution as a bundled dependency; checkpoint terms unknown). |
| Piano transcription | Transkun (https://github.com/Yujia-Yan/Transkun) | Semi-CRF piano AMT with velocity, CPU-capable | Repo MIT (verified). The pip package ships a checkpoint but declares no licence metadata (verified on PyPI JSON: `license: null`); weight terms not stated. | Not bundled; documented candidate adapter. |
| Per-stem neural AMT | Tsumugi | Bass/guitar/vocal/drum variants | Code MIT; checkpoint licence not stated (prior review, MODELS.md). | Not bundled. |
| Drum AMT | ADTOF | Kick/snare/toms/hats/cymbals | CC BY-NC-SA (prior review, MODELS.md). | Excluded. |
| Polyphonic AMT | Basic Pitch | Instrument-agnostic notes with bends | Apache-2.0. | **In use.** |
| Generative / diffusion separation and resynthesis | e.g. Multi-Source Diffusion Models (Mariani et al., ICLR 2024); MIDI-conditioned synthesis | Separation as conditional generation; regeneration of stems | Not verified here. | Not implemented (see §4). |

## 2b. Transcription models evaluated in the research phase (October 23, 2026)

| Model | What was verified | Licence (verified) | Result here |
|---|---|---|---|
| **MuScriptor** (Kyutai + Mirelo, July 2026; https://github.com/muscriptor/muscriptor, paper https://huggingface.co/papers/2607.08168) | Decoder-only transformer; 5 s mel chunks at 16 kHz; 36 instrument groups; optional instrument conditioning (`--instruments`); JSON or MIDI output with a beat-tracked tempo map; **no velocity**; drums onset-only; same-pitch same-instrument notes cannot overlap. Sizes small 103M (CPU-practical), medium 307M, large 1.4B. Model card: onset F1 51.2 / 52.4 / 53.2, multi-instrument F1 38.2 / 39.7 / 40.5 on its 372-track test set; choir frame F1 80.7 vs 51.0 for YourMT3+. | Code MIT. Weights CC BY-NC 4.0 plus conditions (research purposes, rights to the transcribed audio, indemnity); **gated**: each user must accept with their own Hugging Face account. | Adapter implemented and tested with a stub engine. Not run: no Hugging Face login with accepted conditions exists on this machine, and accepting them is the user's decision. |
| **YourMT3+** YPTF.MoE+Multi (noPS) (https://huggingface.co/spaces/mimbres/YourMT3) | Inference reproduced from the authors' `model_helper.py`; checkpoint SHA-256 matches the Hub's LFS id. CPU: about 2.8 s per second of audio. No velocity (all 1). | Code GPL-3.0; checkpoint repo card Apache-2.0. Loader unpickles the checkpoint. | Run on all 20 BabySlakh songs: best transcription measured here (EVALUATION.md). Bass references needed an octave correction to score it fairly. |
| **Transkun** 2.0.1 | Semi-CRF piano model with velocity; checkpoint ships in the pip package; CPU-capable (about 1.5 s per second). | Code MIT; checkpoint licence not stated. | True piano stems: F1 0.97 vs 0.75 (Basic Pitch). Separated piano in dense mixes: 0.43 vs 0.34. |
| Omnizart, ADTOF, Tsumugi | See MODELS.md: stalled TensorFlow stack (Omnizart), CC BY-NC-SA weights (ADTOF), unstated checkpoint licence (Tsumugi). | – | Not integrated in this phase. ADTOF would be permissible for non-commercial research but its weights are TensorFlow; time went to the multi-instrument models, which showed the largest measured gains. |

Benchmark data: BabySlakh 16 kHz (https://zenodo.org/records/4603870, CC BY 4.0), 20 songs with every
instrument's audio and MIDI. Renderer: FluidSynth 2.6.1 with MuseScore General (MIT).

## 3. How a Suno-like system could work, and what that implies

Suno's own description ("listens to it and regenerates each stem") is consistent with a
conditional generative model: an encoder that analyses the mix (and possibly the target
instrument), and a decoder that synthesises the target stem, trained on paired multitrack data at
very large scale. The public literature contains each piece:

1. **Source identification**: tagging/embedding models (PANNs, CLAP-style).
2. **Target conditioning**: a class or query embedding injected into a separator (Banquet, AudioSep).
3. **Estimation**: masking separators (Demucs, RoFormer) estimate the target's contribution;
   generative separators sample it.
4. **Refinement / regeneration**: diffusion or codec-language models resynthesise a clean stem,
   optionally guided by the masked estimate or by a transcription (MIDI-conditioned synthesis).

What follows for an open-source build:

* **Reproducible now, with permissive weights:** steps 1 and 3 (PANNs + HTDemucs), plus
  evidence-gated targeted splits with existing models (6-stem ratios, harmonic/percussive masks).
* **Reproducible only with non-commercial weights:** query-conditioned music separation
  (Banquet). It would be the right engine for "Split from Mix" in a research build.
* **Requires training:** a commercially usable instrument-conditioned separator covering ~100
  classes, or a regenerative stem model. Training data (licensed multitracks at scale) is the
  real barrier, not code.
* **Proprietary:** Suno's models, data and conditioning scheme.

There is also a product-level trade-off. Regenerated stems can sound cleaner while being less
faithful: they need not sum to the mix, and a generated note is not evidence that the performer
played it. Stemforge's goal (faithful MIDI, stems that explain the original mix) favours masking
separation with reconstruction checks. Regeneration is out of scope until it can be measured
against that goal.

## 4. Recommendation (implemented)

1. Keep HTDemucs as the broad separator. Fix its DC/hiss floor and gate it where the mixture is
   digitally silent (done; see EVALUATION.md).
2. Add **PANNs Cnn14** (MIT + CC BY 4.0) as instrument evidence and build discovery on two
   independent signals: activity relative to the mix, and tags computed on each candidate.
3. Treat "Other" as a residual to explain. Run targeted, *mask-based* splits only when tags justify
   them, so that target + complement == parent and nothing is invented. Keep a split only if the
   target's own tags confirm it. Name what remains from evidence, or mark it unresolved.
4. Make query-conditioned separation (Banquet) and stronger AMT (Transkun, YourMT3+) adapter
   targets for users who accept their licences. Do not bundle them.
5. Do not ship regeneration. Report reconstruction error so users can see how well the stems
   explain the recording.
