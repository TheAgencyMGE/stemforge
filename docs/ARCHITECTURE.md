# Stemforge: Architecture and Model Selection

Stemforge is a local-first studio for taking a song apart. It splits the mix into stems, transcribes each stem to MIDI, and lets you edit and export the result.

This document records what was researched, what was chosen, and why. It was written before implementation and updated afterwards wherever measurements changed a decision (see `docs/EVALUATION.md`). Wider instrument coverage, and the models that were reviewed and left out, are in [MODELS.md](MODELS.md). The same decisions are served at `/api/instruments`.

## 1. Research summary

| Project | What it gives us | License (code / weights) | Status (Oct 2026) | Decision |
|---|---|---|---|---|
| **Demucs v4 / HTDemucs** (adefossez/demucs) | 4-stem (`htdemucs`, `htdemucs_ft`) and 6-stem (`htdemucs_6s`: + guitar, piano) separation. Strong SDR, plain PyTorch. | MIT / weights are hosted by Meta with no separate weight license; treat as "MIT-repo, verify for commercial use". | Officially maintained fork, but the author states no new features are planned. | **Default separator.** `htdemucs` for the 4-stem pass. `htdemucs_6s` is optional for guitar and piano. The README itself says 6s piano has "a lot of bleeding and artifacts", so the UI warns about it. |
| **python-audio-separator / UVR** (uziproj fork of nomadkaraoke) | MDX-Net, VR, Roformer, and Demucs models behind one `Separator` API. BS/Mel-Roformer vocals beat HTDemucs on vocals. | MIT code. Most UVR community weights have **no stated license**. | Active upstream; the fork has no releases. | **Optional adapter** (`separator.engine=audio_separator`). Disabled by default because the weight licenses are unverifiable, and the package pulls heavy, unpinned model downloads. |
| **Basic Pitch** (spotify/basic-pitch) | Lightweight polyphonic AMT that is instrument-agnostic. Gives onsets, frames, contours, and pitch bends. Runs on ONNX/TFLite/CoreML/TF. | Apache-2.0; model files ship inside the package. | Python ≤3.11; last release 0.4.0. | **Default polyphonic engine** (piano, guitar, other, and a bass fallback). Used through ONNX Runtime with `--no-deps` to avoid TensorFlow, which caps numpy and adds about 1.5 GB. |
| **basic-pitch-ts** | The same model in TF.js for the browser. | Apache-2.0 | Maintained | Not used. Server-side inference keeps a single source of truth and avoids shipping 20 MB of TF.js. |
| **Band Transcriber** (VanKyle00) | Reference pipeline: htdemucs_6s, then librosa onset drums, basic-pitch for bass and vocals, and MT3 for guitar and piano on Modal. | Not stated | 0 stars, hobby project | Confirms the overall pipeline shape. No code was reused because the license is unknown. Its own README calls polyphonic guitar and piano "experimental". |
| **Tsumugi** (anime-song) | Neural Semi-CRF AMT (Transkun-derived) with per-stem variants (bass, guitar, vocals, drums), plus velocity and instrument refinement. | MIT code / **checkpoint license not stated**; the SheetSage beat model is CC BY-NC 4.0. | Very active (changelog 2026-10-17) | Promising, but excluded from the default build. The weight license is not stated, it recommends 12 GB+ VRAM (the dev box has 8 GB), and it needs torch 2.13. It is documented as a future adapter target in `TranscriptionEngine`. |
| **MT3 / YourMT3** | Multi-instrument seq2seq AMT. MT3 is JAX/T5X; YourMT3 is PyTorch. | Apache-2.0 code; MT3 checkpoints are Apache-2.0 | MT3 is archived and version-fragile | Not integrated. The JAX stack is impractical on Windows. YourMT3 is a candidate adapter. |
| **librosa** | Beat tracking, onset detection, pYIN f0, chroma, CQT. | ISC | Active | **Used** for analysis, the monophonic engine, and drums. |
| **madmom** | Better beat and downbeat tracking. | BSD code / **CC BY-NC-SA models** | Py3.11+ builds broken | Rejected: the model license and the build status both rule it out. |
| **ADTOF / Omnizart drums** | Neural drum transcription. | ADTOF models CC BY-NC-SA; Omnizart stalled on TF 2.5 | Stale | Rejected for the default build. Drums use semi-adaptive NMF (Dittmar & Gärtner 2014), implemented here in numpy (below). |
| **mido / pretty_midi / symusic** | SMF writing and parsing. | MIT / MIT / MIT | Active | We **write with mido**. Every file is **re-parsed with symusic** (an independent C++ parser) and checked by the frontend tests with `@tonejs/midi`. |

## 2. Pipeline

```
upload ─► validate (magic bytes; header probe via libsndfile, or PyAV/ffmpeg for M4A; size/duration/channel limits)
       ─► decode → 44.1 kHz float32 stereo master (+ optional trim window)
       ─► analyze  : duration, peak/clipping, tempo + beat grid, key (Krumhansl–Schmuckler),
                     sections (chroma/MFCC agglomerative segmentation)
       ─► separate : htdemucs (vocals, drums, bass, other)   [+ htdemucs_6s → guitar, piano]
                     residual check: |mix − Σ stems| reported as separation leakage metric
       ─► transcribe (per stem, engine chosen by policy below)
       ─► MIDI     : per-stem SMF + combined multitrack SMF (tempo map from analysis),
                     re-parsed with symusic for validation
       ─► artifacts: stems (FLAC 24-bit), lossless 16-bit FLAC previews (no MP3/Opus encoder delay,
                     so nothing is silently shifted), peaks JSON, constant-Q pitch spectrogram PNG
```

Every stage writes its output to storage and records an `Artifact` row. A retried or resumed job therefore **skips stages whose artifacts already exist** and still match the job's parameter hash.

### Transcription engine policy (explainable)

| Stem | Engine | Why |
|---|---|---|
| vocals | `pyin-melody`: pYIN f0 → note segmentation, plus pitch bends from the f0 contour | Vocals are mostly monophonic. pYIN gives a calibrated voiced probability, which we use directly as per-note confidence. Basic Pitch over-splits vibrato. |
| bass | `basic-pitch` band-limited to 30–500 Hz, onset 0.6; `pyin` one click away | Planned as pYIN. Measurement reversed it: Basic Pitch F1 0.91 vs pYIN 0.79, because pYIN merges repeated same-pitch notes. pYIN remains better for legato lines. |
| piano, guitar, other | `basic-pitch` (ONNX) | Polyphonic transcription is required. Confidence is the mean note activation. |
| drums | `drum-nmf`: semi-adaptive NMF with kick, snare and hi-hat templates (GM 36/38/42). A hit is relabeled open hat, pedal hat, crash, tom, side-stick or clap only when its spectrum disagrees with that template. | No permissively licensed neural drum model exists (ADTOF is CC BY-NC-SA). A band-flux prototype over-triggered (F1 0.49–0.72). NMF separates simultaneous hits (F1 1.0 on the fixture for the three coarse classes). |

The user can override the engine per track and re-run transcription only for that track, using sensitivity (onset/frame thresholds) and a minimum note length. Separation is not re-run.

### Confidence

Confidence comes from each engine's own probabilities. It is never invented.
- `basic-pitch`: mean of the model's note-activation over the note span.
- `pyin`: mean voiced probability over the note.
- `drum-nmf`: how far the activation onset rises above its adaptive peak-picking threshold, as a fraction of the peak.
- Notes the user draws have `confidence: null`. They are never given a fabricated score.
- Stems below −55 dBFS, or more than 40 dB under the mix, are reported silent and not transcribed. Stems more than 20 dB under the mix get a bleed warning.

The piano roll shades notes by confidence. Notes below a user-set threshold are outlined as "uncertain". Stems from `htdemucs_6s` piano are labelled "low separation reliability", with a link to the upstream caveat.

## 3. System architecture

```
 Next.js (React 19, TS)  ──REST/JSON──►  FastAPI (uvicorn)  ──SQL──►  SQLite | Postgres
   DAW workspace, piano roll,              validation, projects,           projects, jobs,
   Web Audio sync engine                   jobs, signed downloads          stages, artifacts
                                                 │                              ▲
                                                 ▼ storage (local FS | S3)      │ lease/heartbeat
                                           Worker process(es) ──────────────────┘
                                           GPU/CPU ML inference, one job per worker slot
```

* **Task queue.** The queue is backed by the database (`jobs` table) with leases, heartbeats, a capped exponential-backoff retry policy, cancellation flags, and stale-lease recovery. SQLite claims jobs with an atomic conditional `UPDATE`; Postgres uses `FOR UPDATE SKIP LOCKED`.
  * *Why not Celery/RQ:* those need Redis, which isn't natively available on Windows, the primary dev target. They would also add a second source of truth for job state. The DB queue gives durable state, a resumable stage history, and the same semantics in dev and production.
  * Workers are separate processes started with `stemforge-worker`. You can run N of them, and they can run on GPU hosts apart from the API.
* **Storage.** The `Storage` protocol has two implementations: `LocalStorage` (a private directory, paths resolved and confined under the root) and `S3Storage` (private bucket, presigned GETs). The API never serves the storage root statically. Downloads go through HMAC-signed, expiring URLs (local) or presigned URLs (S3).
* **DB.** The database layer is SQLAlchemy 2 with Alembic migrations. Pydantic v2 schemas form the typed API contract, published at `/api/openapi.json`. The frontend mirrors them by hand in `lib/types.ts`; the end-to-end tests exercise the contract.
* **Frontend audio.** One `AudioContext` drives playback. All stems are decoded into `AudioBuffer`s and started with the *same* `when` time, which gives sample-accurate sync. Each lane has a gain node and a solo/mute matrix. MIDI preview uses a small built-in Web Audio synth that is scheduled against the same clock, so audio and MIDI are compared on one timeline.
* **Editing.** Notes live in a client store with an undo/redo command stack and debounced autosave of the edit document (`PUT /projects/{id}/tracks/{t}/notes`). MIDI export is rendered **server-side** from the edited notes, so the exported file always goes through the same writer and validator.

## 4. Security and operations

* **Uploads.** Requests are streamed to disk with a byte cap. The container is sniffed from magic bytes (RIFF/WAVE, fLaC, ID3/MPEG sync, OggS, ftyp, and others) rather than trusted from the extension. The file is then fully decoded in the worker with limits on duration, sample rate, and channel count. Stored names are random UUIDs; the original filename is sanitised and kept only as metadata.
* **Limits and rate limiting.** Uploads are rate-limited per client IP with a token bucket. Each instance has a storage quota. Retention cleanup deletes projects after `RETENTION_DAYS`, and users can delete projects themselves.
* **Logs and health.** Logs are structured JSON. `/healthz` reports liveness and `/readyz` checks the DB, storage, and model files. Errors carry a request id.
* **Models.** `stemforge-models download` fetches only manifest-listed URLs, verifies their SHA-256 before the file is moved into the model directory, and refuses anything else. The loader re-verifies the hash before any `torch.load`, because Demucs checkpoints are pickles.

## 5. Known limitations (honest)

* Separation quality is bounded by HTDemucs. Reverb tails and dense mixes bleed between stems.
* Six-stem piano is unreliable upstream. Guitar is usable.
* Polyphonic transcription of distorted guitar or dense synths is ambiguous and will show low confidence.
* Drum transcription starts from kick, snare, and hi-hat templates. Open hat, pedal hat, crash, tom, side-stick and clap are spectral relabels, not a neural model. Ride, tambourine and shaker are not detected. Optional kit audio is an NMF mask of the drum stem, not a new separator.
* M4A/AAC decoding needs FFmpeg (PyAV or the binary). The primary decoder is libsndfile (WAV, FLAC, OGG/Opus, MP3, AIFF), because PyAV's DLLs can be blocked by Windows Smart App Control.
* Postgres, S3 and the Docker images are implemented but were not executed on the development machine.
* Key detection is the global key only, and section labels are unsupervised (A/B/C), not "verse/chorus".
