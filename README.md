# Stemforge

Stemforge takes a song apart. Upload a track and it runs a fixed pipeline:

1. **Separate** the mix into vocals, drums, bass and "other", and optionally guitar and piano.
2. **Transcribe** each stem to MIDI with an engine chosen for that kind of instrument.
3. **Edit** the result in a synced, DAW-style workspace: multitrack audio, a piano roll over a pitch spectrogram, a confidence overlay, quantize, snap-to-key and velocity editing.
4. **Export** stems, per-track MIDI, a multitrack MIDI file, or a restorable project bundle.

All inference is real:
- **Separation:** HTDemucs.
- **Pitched transcription:** Spotify Basic Pitch and pYIN.
- **Drums:** semi-adaptive NMF.

Nothing is simulated. Silent or near-silent stems are reported as such and are not transcribed.

```
Upload → Analyze → Separate → Transcribe → Edit → Export
```

* Architecture and model selection: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* Measured accuracy on the bundled fixture: [docs/EVALUATION.md](docs/EVALUATION.md)
* Licenses and attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Features

| Area | What you get |
|---|---|
| **Pipeline** | Validation by magic bytes and a full decode. Analysis: tempo with a regression-refined beat grid, key with a runner-up, and unsupervised sections. HTDemucs 4-stem separation, plus optional `htdemucs_6s` for guitar and piano. Per-stem transcription, then MIDI written with mido and re-parsed by symusic. |
| **Engines** | **Basic Pitch:** polyphonic (piano, guitar, other, and an optional pass on the whole mix) and band-limited bass, with pitch bends. **pYIN:** monophonic, the vocal default, with bends from the f0 contour. **Drum NMF:** kick, snare and hi-hat (GM 36/38/42), with spectral relabels for open hat, pedal hat, crash, tom, side-stick and clap when the hit does not match the template. The policy is explained in the UI, and you can switch engine per track. |
| **Re-tuning** | Onset, sustain and voicing thresholds, minimum note length and frequency band. Model output is cached, so re-thresholding takes about a second. |
| **Workspace** | Sample-locked Web Audio playback of every stem. Per-lane mute, solo, volume, audio monitor and MIDI-preview synth. A/B against the original mix. Loop regions, a follow playhead, and bar/beat ruler with section markers. |
| **Piano roll** | Move, resize (either edge), insert, delete and marquee-select notes, with copy/paste and duplicate. Snap to 1/1–1/32 including triplets. Quantize with strength, transpose, snap-to-key, a velocity lane, a bend display, and a constant-Q spectrogram underlay aligned to pitch rows. The confidence overlay dashes uncertain notes; user-drawn notes are marked as such. |
| **Editing safety** | Per-track undo/redo, debounced autosave with optimistic concurrency (409 on conflicts), and reset to the model output. |
| **Export** | One-click full export, a selective zip, multitrack or per-track MIDI built from your edits, and a full project bundle that you can re-import. Trimmed sections keep the original timeline: MIDI is offset accordingly. |
| **Jobs** | Durable DB queue with leases, heartbeats, backoff retries and cancellation. Resume skips finished stages through artifact hashes. Stage-level progress, and an ETA only once it is measurable. |
| **Ops** | Private storage (local or S3) behind HMAC-signed or presigned URLs. Size, duration and channel limits, a storage quota, retention cleanup and rate limiting. Structured JSON logs with request ids, `/api/healthz` and `/api/readyz`, Alembic migrations, and an optional API token. |

## Quick start (local, Windows)

Requirements: Windows 10/11, Node 20+, and either an NVIDIA GPU with a recent driver or a CPU. `winget` is used to install `uv`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1        # add -Cpu for CPU-only PyTorch
powershell -File scripts\dev.ps1                                   # starts API, worker and web UI
```

Open http://localhost:3000 and press **Try the demo song**.

Manual equivalent:

```powershell
uv venv --python 3.11 $env:LOCALAPPDATA\stemforge\venv
$py = "$env:LOCALAPPDATA\stemforge\venv\Scripts\python.exe"
uv pip install --python $py torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $py --no-deps -r backend\requirements\lock.txt
uv pip install --python $py --no-deps -e backend
cd backend; stemforge-models download; stemforge-api     # terminal 1 (http://127.0.0.1:8000)
cd backend; stemforge-worker                              # terminal 2
cd frontend; npm ci; npm run dev                          # terminal 3 (http://localhost:3000)
```

## Quick start (Linux / macOS)

```bash
./scripts/setup.sh          # or ./scripts/setup.sh --cpu
```

The script then prints the three commands that start the API, worker and web UI. On macOS, PyTorch uses the CPU here (MPS is not wired up).

## Docker

```bash
cp .env.example .env      # set STEMFORGE_SIGNING_SECRET (e.g. `openssl rand -hex 32`)
docker compose up --build                                            # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA GPU (Container Toolkit required)
```

Compose runs these services:

| Service | Role |
|---|---|
| `db` | Postgres 17 |
| `models` | One-shot, verified weight download into a volume |
| `api` | FastAPI |
| `worker` | ML inference |
| `web` | Next.js standalone, bound to `127.0.0.1:3000` |

> **Not verified:** the Docker images and the Postgres code path were written for this release but have not been built or run on the development machine, which has no Docker. CI builds and tests the same commands on Linux.

## Model downloads (verified, no pickles)

`stemforge-models download [htdemucs htdemucs_6s htdemucs_ft]` fetches only files listed in [`backend/stemforge/model_manifest.json`](backend/stemforge/model_manifest.json):
- **Pinned source:** each file comes from the `adefossez/HTDemucs*` Hugging Face repo at a fixed commit.
- **Verified:** each file is SHA-256 checked against the Hub's LFS object id before it is moved into place, and checked again before every load.
- **No pickles:** only `.safetensors` are accepted, and only classes under `demucs.` are instantiated.

Defaults are `htdemucs` (84 MB) and `htdemucs_6s` (55 MB). `htdemucs_ft` (4 × 84 MB, about 4× slower) is optional; select it with `STEMFORGE_DEMUCS_MODEL=htdemucs_ft`. Basic Pitch's ONNX model ships inside the `basic-pitch` package.

```bash
stemforge-models list     # status + license notes
stemforge-models verify   # re-hash installed files
```

## GPU and CPU

* **GPU.** Install the CUDA 12.8 wheels (`--index-url https://download.pytorch.org/whl/cu128`). They support RTX 50-series (Blackwell) and older cards. HTDemucs needs about 3 GB of VRAM with the default segment. Set `STEMFORGE_DEMUCS_SEGMENT=4` on small GPUs. A CUDA out-of-memory error automatically retries with smaller segments, then on the CPU.
* **CPU.** Install the CPU wheels or set `STEMFORGE_DEVICE=cpu`. Transcription is always CPU: ONNX Runtime and librosa.
* **Measured on the dev box** (31.5 s demo):

  | | GPU (RTX 5060 8 GB) | CPU only (12 threads) |
  |---|---|---|
  | htdemucs separation | 1.5 s | 13.2 s (≈0.42× real time) |
  | Whole job | ≈10 s | – |
  | Full-pipeline test | – | 37 s, passing |

  Mix residual is the same on both devices (−28.8 dB).

## Configuration

Every setting is an environment variable `STEMFORGE_<NAME>` (or a line in `backend/.env`). See [`backend/.env.example`](backend/.env.example) and [`stemforge/config.py`](backend/stemforge/config.py).

| Variable | Default | Notes |
|---|---|---|
| `DATA_DIR` | `backend/data` | SQLite DB plus private storage. Keep it short on Windows. |
| `DATABASE_URL` | sqlite in `DATA_DIR` | e.g. `postgresql+psycopg://…` |
| `STORAGE_BACKEND` | `local` | `s3` with `S3_BUCKET`, `S3_PREFIX`, `S3_ENDPOINT_URL`, `S3_REGION` (needs `boto3`) |
| `SIGNING_SECRET` | generated file | **Required in production.** HMAC key for download links. |
| `API_TOKEN` | empty | Bearer token for every `/api` route. The UI prompts for it. |
| `MAX_UPLOAD_MB` / `MAX_DURATION_SECONDS` | 200 / 900 | Longer files can be trimmed to a section at upload. |
| `STORAGE_QUOTA_GB` / `RETENTION_DAYS` | 50 / 14 | `RETENTION_DAYS=0` disables auto-deletion. |
| `RATE_LIMIT_PER_MINUTE` | 30 | Mutating requests per IP. An upload costs 5. |
| `DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `DEMUCS_MODEL` / `DEMUCS_SEGMENT` / `DEMUCS_SHIFTS` | `htdemucs` / auto / 0 | `shifts>0` adds random time shifts: slower, and not deterministic. |
| `WORKER_CONCURRENCY` | 1 | Processes per `stemforge-worker`. You can also run more workers. |
| `JOB_MAX_ATTEMPTS` / `JOB_LEASE_SECONDS` | 3 / 120 | Retry and lease policy |

## Production deployment

1. **Database and storage.** Run Postgres and S3 (or S3-compatible) storage. Set `DATABASE_URL`, `STORAGE_BACKEND=s3` and a long random `SIGNING_SECRET`. Migrations run automatically on API and worker start; `stemforge-migrate` runs them explicitly.
2. **Topology.** Run the API on CPU nodes and workers on GPU nodes. They share only the DB and storage. Scale workers horizontally: jobs are claimed with `FOR UPDATE SKIP LOCKED`.
3. **Edge.** Put the web app and `/api` behind a TLS reverse proxy. For multi-user deployments add real authentication there (an OAuth2/OIDC proxy). The built-in `API_TOKEN` is a single shared secret meant for private single-tenant installs.
4. **Request size.** Raise the proxy's body-size limit to at least `MAX_UPLOAD_MB`.
5. **Monitoring and retention.** Monitor `/api/readyz`, and ship the JSON logs. Retention cleanup runs in every worker every 10 minutes; `stemforge-cleanup` runs it once.

## Tests

```bash
cd backend && pytest -q                    # 44 tests: units, queue, API, full real-model pipeline
cd frontend && npm test                    # 25 unit tests (vitest)
cd frontend && npm run test:e2e            # Playwright; needs stemforge-api + stemforge-worker running
python backend/scripts/evaluate.py --out docs/EVALUATION.md   # accuracy vs ground truth
```

Also: `ruff check`, `mypy stemforge`, `npm run lint`, `npm run typecheck`, `npm run build`. CI runs all of these ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Sample audio

`samples/stemforge-demo.flac` is a 31.5 s song (A minor, 96 BPM) synthesised from scratch by [`samples/generate_demo.py`](samples/generate_demo.py). It is dedicated to the public domain (CC0-1.0). Its exact score is in `stemforge-demo.truth.json`, which makes the demo an evaluation fixture as well.

## Known limitations

* **Separation.** Quality is bounded by HTDemucs: reverb and dense mixes bleed between stems. The 6-stem piano is unreliable, as upstream notes; guitar is usable. The synthetic demo's lead synth lands in "Other", so its vocal stem is (correctly) silent.
* **Polyphonic transcription.** Dense or distorted material stays ambiguous. On the fixture the "Other" stem reaches F1 0.66: it finds every note, but adds extra ones (octave and harmonic errors). Use the confidence overlay and the cleanup tools.
* **Drums.** The learned part is three NMF templates. Open hat, pedal hat, crash, tom, side-stick and clap are relabels when the spectrum disagrees; ride, tambourine and shaker are not detected. NMF is near perfect on the synthetic kit's kick, snare and closed hat; expect lower accuracy on real kits with heavy room sound. Strings, brass, woodwinds and most keyboards are not separated. See [docs/MODELS.md](docs/MODELS.md).
* **Analysis.** Key is global Krumhansl–Schmuckler, and relative major/minor can swap (the demo reads C major, runner-up A minor). Tempo is one constant tempo; the beat grid has no tempo map. Section labels are unsupervised letters.
* **MIDI pitch bends.** These are only emitted for monophonic passages, because bend is per channel.
* **Browser memory.** Every stem is decoded to float PCM, roughly 60 MB per stem per 3 minutes.
* **Formats.** M4A/AAC needs FFmpeg (PyAV or the `ffmpeg` binary). Without it the server rejects those files with an explanatory error.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: torch._strobelight…`, or `WinError 206` / "filename too long" | Windows MAX_PATH. Keep the venv and `STEMFORGE_DATA_DIR` on short paths (the setup script does), or enable `LongPathsEnabled`. |
| `DLL load failed … Application Control policy has blocked this file` | Windows Smart App Control blocked a native wheel. On first install this is often temporary, while the file's reputation is checked. Stemforge falls back to libsndfile when PyAV is blocked. Don't disable OS security; retry later, or use WSL/Docker. |
| UI says "Can't reach the Stemforge API" | Start `stemforge-api` (port 8000). The web app proxies `/api` to `STEMFORGE_API_URL`. |
| Jobs stay "Waiting for a worker" | Start `stemforge-worker`. Check its log for the device line. |
| "separation model unavailable" | Run `stemforge-models download`. |
| "SHA-256 mismatch" | Delete the file under `backend/models/demucs/<model>/` and download it again. |
| CUDA out of memory | Set `STEMFORGE_DEMUCS_SEGMENT=4`. Close other GPU apps. It falls back to the CPU automatically. |
| "Conflict" save state | The track changed elsewhere (another tab). Reload the page. |
| 401 everywhere | `STEMFORGE_API_TOKEN` is set. Enter it in the prompt the UI shows. |
