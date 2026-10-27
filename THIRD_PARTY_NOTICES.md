# Third-party notices and model licensing

Stemforge's own code is MIT-licensed. It depends on the projects below. **No model weight used here carries an explicit commercial-use grant from its authors.** Check the status of each one yourself before you ship commercially.

## Models and weights

| Component | Source | Code license | Weights |
|---|---|---|---|
| HTDemucs (`htdemucs`, `htdemucs_6s`, `htdemucs_ft`) | https://github.com/adefossez/demucs; weights from `huggingface.co/adefossez/HTDemucs*` at pinned commits | MIT | The weight repositories state no license. The maintainer has said the weights are not covered by MIT and are provided for scientific purposes (facebookresearch/demucs#327). `htdemucs` was trained by Meta AI on MUSDB18-HQ plus internal data. For `htdemucs_6s` the author said the training data is Meta's and gave no commercial answer (demucs#418). Treat commercial use as unverified. |
| Basic Pitch (ICASSP 2022) | https://github.com/spotify/basic-pitch | Apache-2.0 | The model files ship inside the Apache-2.0 package, and the repo states no separate weight license. |
| pYIN | librosa (`librosa.pyin`) | ISC | No learned weights |
| PANNs Cnn14 (16 kHz), instrument tagging for source discovery | Kong, Cao, Iqbal, Wang, Wang, Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM TASLP 2020. Code: https://github.com/qiuqiangkong/audioset_tagging_cnn. Weights: `Cnn14_16k_mAP=0.438.pth`, https://zenodo.org/records/3987831 | MIT (Copyright (c) 2018-2020 Qiuqiang Kong). Stemforge re-implements the forward pass in `stemforge/pipeline/tagger.py`. | **CC BY 4.0** (attribution required; commercial use permitted under that licence). Converted to safetensors by `stemforge-models`; tensors unchanged. Class names come from Google's AudioSet ontology (CC BY 4.0). |
| Drum NMF | Stemforge (`stemforge/pipeline/drums.py`, `transcription.py`), after Dittmar & Gärtner, DAFx 2014 | MIT | Fixed templates defined in code. No ADTOF weights. Spectral relabels (open hat, tom, crash, and so on) are rules in this repo, not a checkpoint. |

## Optional research components (not installed by default, each isolated)

These are installed only by `scripts/setup-research.ps1`, each in its own environment or folder outside
the repository, and are called as separate programs. Stemforge does not import their code or weights
into its own process and does not redistribute them. They are for **non-commercial research**.

| Component | Use in Stemforge | Code licence | Weights / data | Restrictions |
|---|---|---|---|---|
| Transkun 2.0.1 (Yan & Duan) | Piano transcription with velocity (`transkun` engine) | MIT | Checkpoint ships in the pip package with no stated licence | Treat as research-only. |
| YourMT3+ YPTF.MoE+Multi (Chang et al., 2024), https://huggingface.co/spaces/mimbres/YourMT3 | Multi-instrument transcription (`yourmt3` engine) | GPL-3.0 (run as a separate program, not linked) | https://huggingface.co/mimbres/YourMT3, card says Apache-2.0; SHA-256 pinned in the setup script | Its loader unpickles the checkpoint (`weights_only=False`); this runs only in the isolated environment. |
| MuScriptor 0.4.0 (Kyutai and Mirelo, 2026), https://github.com/muscriptor/muscriptor | Multi-instrument transcription (`muscriptor` engine) | MIT | **CC BY-NC 4.0** plus the model card's conditions (research purposes, rights to the transcribed audio, indemnity); gated on Hugging Face | Non-commercial only. Each user must accept the conditions with their own Hugging Face account; Stemforge never does it for them. |
| FluidSynth 2.6.1, https://github.com/FluidSynth/fluidsynth | Rendering MIDI for fidelity measurements | LGPL-2.1 (separate program) | – | – |
| MuseScore General SF3 v0.2 | General MIDI sounds for rendering | – | MIT; derived from FluidR3 (Frank Wen), FluidR3Mono (Michael Cowgill), adaptation by S. Christian Collins; Temple Blocks by Ethan Winer; Drumline Cymbals by Michael Schorsch | Keep these acknowledgements with any derivative. |

## Evaluation data (not in the repository)

| Dataset | Use | Licence |
|---|---|---|
| BabySlakh 16 kHz (Manilow, Wichern, Seetharaman, Le Roux, "Cutting Music Source Separation Some Slakh", WASPAA 2019), https://zenodo.org/records/4603870 | Separation, transcription and render benchmarks (`backend/scripts/evaluate_slakh.py`) | CC BY 4.0 (MD5 `311096dc2bde7d61c97e930edbfc7f78` checked) |

Evaluated but **not bundled**:

| Project | Reason |
|---|---|
| python-audio-separator / UVR models | Most weights have no stated license. An optional adapter exists. |
| Banquet / query-bandit (query-conditioned separation) | Code MIT; checkpoints CC BY-NC-SA 4.0 (Zenodo 13694558). |
| AudioSep (text-queried separation) | Code MIT; checkpoint licence not stated. |
| YourMT3 / YourMT3+ | Code GPL-3.0; checkpoint licence and hosting not stated. |
| Transkun (piano) | Repo MIT; the pip package declares no licence and the shipped checkpoint's terms are not stated. |
| LAION-CLAP | Code CC0-1.0; checkpoint licence not stated; part of the training data is research-only. |
| DrumSep and other MSST community checkpoints | No weight licence stated. |
| Tsumugi | Checkpoint license not stated; its SheetSage beat model is CC BY-NC 4.0. |
| MT3 | JAX stack |
| madmom models | CC BY-NC-SA |
| ADTOF | CC BY-NC-SA |

## Libraries (selection)

* **Python:** PyTorch (BSD-3), torchaudio (BSD-2), FastAPI (MIT), Starlette (BSD-3), SQLAlchemy (MIT), Alembic (MIT), Pydantic (MIT), librosa (ISC), NumPy/SciPy (BSD-3), scikit-learn (BSD-3), soundfile (BSD-3) with libsndfile (LGPL-2.1, dynamically linked), soxr (LGPL-2.1), ONNX Runtime (MIT), mido (MIT), pretty_midi (MIT), symusic (MIT), mir_eval (MIT), safetensors (Apache-2.0), PyAV (BSD-3) with FFmpeg (LGPL/GPL depending on the build), Pillow (MIT-CMU), httpx (BSD-3).
* **JavaScript:** Next.js (MIT), React (MIT), zustand (MIT), @tonejs/midi (MIT, tests only), Playwright (Apache-2.0, tests only), Vitest (MIT).

The full pinned Python set is in `backend/requirements/lock.txt`. JS versions are in `frontend/package-lock.json`.

## Sample audio

`samples/stemforge-demo.flac` and `stemforge-demo.truth.json` were generated by `samples/generate_demo.py`. They are released under CC0-1.0. No third-party recordings or samples were used.

## Your content

You are responsible for having the rights to the recordings you upload, and to anything you derive from them.
