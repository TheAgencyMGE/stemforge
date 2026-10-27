"""Audio decoding, resampling, writing and visual summaries (peaks, spectrogram)."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 44100


class AudioDecodeError(Exception):
    """Raised for files that cannot be decoded safely; message is shown to the user."""


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    duration: float
    frames: int
    decoder: str


def probe(path: Path, fmt: str) -> AudioInfo:
    """Read container metadata without decoding the whole file."""
    if fmt in {"wav", "flac", "ogg", "aiff", "mp3", "caf"}:
        try:
            info = sf.info(str(path))
            if info.frames <= 0 or info.samplerate <= 0:
                raise AudioDecodeError("file contains no audio frames")
            return AudioInfo(info.samplerate, info.channels, info.frames / info.samplerate, info.frames, "libsndfile")
        except RuntimeError as exc:
            if fmt != "mp3":
                raise AudioDecodeError(f"could not read {fmt.upper()} header: {exc}") from exc
    av = _try_import_av()
    if av is not None:
        try:
            with av.open(str(path), metadata_errors="ignore") as c:
                streams = [s for s in c.streams if s.type == "audio"]
                if not streams:
                    raise AudioDecodeError("no audio stream found")
                s = streams[0]
                dur = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / 1e6
                sr = s.codec_context.sample_rate or s.rate
                return AudioInfo(int(sr), int(s.codec_context.channels), dur, int(dur * sr), "pyav")
        except AudioDecodeError:
            raise
        except Exception as exc:
            raise AudioDecodeError(f"could not read {fmt.upper()} container: {exc}") from exc
    if _ffmpeg() is not None:
        return _ffprobe(path)
    raise AudioDecodeError(
        f"{fmt.upper()} needs FFmpeg (PyAV or ffmpeg binary), which is not available on this server. "
        "Convert the file to WAV, FLAC, MP3 or OGG and upload again."
    )


def decode(path: Path, fmt: str, start: float = 0.0, end: float | None = None,
           target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode to float32 array of shape (2, n) at target_sr. Mono is duplicated, >2 channels downmixed."""
    audio, sr = _decode_native(path, fmt)
    if audio.ndim == 1:
        audio = audio[None, :]
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if start > 0 or end is not None:
        s = max(0, int(round(start * sr)))
        e = audio.shape[1] if end is None else min(audio.shape[1], int(round(end * sr)))
        if e <= s:
            raise AudioDecodeError("trim range is empty")
        audio = audio[:, s:e]
    audio = to_stereo(audio)
    if sr != target_sr:
        audio = soxr.resample(audio.T, sr, target_sr, quality="HQ").T
    return np.ascontiguousarray(audio, dtype=np.float32)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    ch = audio.shape[0]
    if ch == 1:
        return np.repeat(audio, 2, axis=0)
    if ch == 2:
        return audio
    # ITU-ish downmix for surround: L,R,C,LFE,Ls,Rs ordering assumed (WAVE default).
    left = audio[0] + 0.707 * (audio[2] if ch > 2 else 0) + 0.707 * (audio[4] if ch > 4 else 0)
    right = audio[1] + 0.707 * (audio[2] if ch > 2 else 0) + 0.707 * (audio[5] if ch > 5 else 0)
    out = np.stack([left, right])
    peak = np.max(np.abs(out)) or 1.0
    return out / max(1.0, peak)


def _decode_native(path: Path, fmt: str) -> tuple[np.ndarray, int]:
    if fmt in {"wav", "flac", "ogg", "aiff", "mp3", "caf"}:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            return data.T, int(sr)
        except RuntimeError as exc:
            if fmt != "mp3":
                raise AudioDecodeError(f"could not decode {fmt.upper()}: {exc}") from exc
    av = _try_import_av()
    if av is not None:
        return _decode_av(av, path)
    if _ffmpeg() is not None:
        return _decode_ffmpeg(path)
    raise AudioDecodeError(f"{fmt.upper()} decoding requires FFmpeg, which is not installed on this server.")


def _try_import_av():
    try:
        import av  # type: ignore

        return av
    except Exception:  # ImportError, or DLL blocked by OS application control
        return None


def _decode_av(av, path: Path) -> tuple[np.ndarray, int]:
    chunks = []
    with av.open(str(path), metadata_errors="ignore") as c:
        stream = next((s for s in c.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioDecodeError("no audio stream found")
        sr = stream.codec_context.sample_rate
        resampler = av.AudioResampler(format="fltp", rate=sr)
        for frame in c.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray())
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray())
    if not chunks:
        raise AudioDecodeError("decoded zero samples")
    return np.concatenate(chunks, axis=1).astype(np.float32), int(sr)


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _ffprobe(path: Path) -> AudioInfo:
    probe_bin = shutil.which("ffprobe")
    if not probe_bin:
        raise AudioDecodeError("ffprobe not available")
    out = subprocess.run(
        [probe_bin, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels:format=duration", "-of", "json", str(path)],
        capture_output=True, timeout=60, check=False,
    )
    if out.returncode != 0:
        raise AudioDecodeError("ffprobe could not read the file")
    meta = json.loads(out.stdout or b"{}")
    streams = meta.get("streams") or []
    if not streams:
        raise AudioDecodeError("no audio stream found")
    sr = int(streams[0]["sample_rate"])
    dur = float(meta.get("format", {}).get("duration", 0))
    return AudioInfo(sr, int(streams[0]["channels"]), dur, int(dur * sr), "ffmpeg")


def _decode_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    info = _ffprobe(path)
    proc = subprocess.run(
        [_ffmpeg() or "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True, timeout=1800, check=False,
    )
    if proc.returncode != 0:
        raise AudioDecodeError("ffmpeg failed to decode the file")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    ch = max(1, info.channels)
    data = data[: (len(data) // ch) * ch].reshape(-1, ch).T
    return data.copy(), info.sample_rate


def write_flac(path: Path, audio: np.ndarray, sr: int = TARGET_SR, subtype: str = "PCM_24") -> None:
    """Write (channels, n) float audio as FLAC. Values are clipped only at write time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio.T, -1.0, 1.0), sr, format="FLAC", subtype=subtype)


_PCM_BITS = {"PCM_16": 16, "PCM_24": 24}


class FlacVerificationError(Exception):
    pass


def quantize(audio: np.ndarray, subtype: str = "PCM_24") -> np.ndarray:
    """The integer samples libsndfile writes for (channels, n) float audio: clip, scale, round."""
    bits = _PCM_BITS[subtype]
    scale = float(2 ** (bits - 1))
    q = np.round(np.clip(audio.astype(np.float64), -1.0, 1.0) * scale)
    return np.clip(q, -scale, scale - 1).astype(np.int32)


def write_flac_verified(path: Path, audio: np.ndarray, sr: int = TARGET_SR, subtype: str = "PCM_24") -> dict:
    """Write FLAC, decode it again, and require the decoded PCM to equal the quantized input exactly.

    FLAC is lossless, so the only change between the float waveform and the file is the documented
    quantization to ``subtype`` (and clipping of samples outside [-1, 1], which is counted). Returns
    facts for an export manifest: format, frames, channels, a hash of the PCM, and clip count.
    """
    import hashlib

    write_flac(path, audio, sr, subtype)
    want = quantize(audio, subtype)
    info = sf.info(str(path))
    got, got_sr = sf.read(str(path), dtype="int32", always_2d=True)
    shift = 32 - _PCM_BITS[subtype]  # libsndfile left-justifies narrower PCM in int32
    got = (got >> shift).T
    if got_sr != sr or got.shape != want.shape or not np.array_equal(got, want):
        raise FlacVerificationError(f"{path.name}: decoded PCM differs from the encoder input")
    return {
        "format": "FLAC", "subtype": subtype, "bit_depth": _PCM_BITS[subtype], "sample_rate": int(info.samplerate),
        "channels": int(info.channels), "frames": int(info.frames),
        "pcm_sha256": hashlib.sha256(np.ascontiguousarray(want.T).tobytes()).hexdigest(),
        "clipped_samples": int(np.count_nonzero(np.abs(audio) > 1.0)), "verified": "decoded PCM == encoder input",
    }


def write_wav(path: Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.T if audio.ndim == 2 else audio
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, format="WAV", subtype=subtype)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def level_stats(audio: np.ndarray) -> dict:
    mono = to_mono(audio)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0
    return {
        "rms_db": round(float(20 * np.log10(max(rms, 1e-9))), 2),
        "peak_db": round(float(20 * np.log10(max(peak, 1e-9))), 2),
        "clipped_ratio": round(clipped, 6),
    }


def compute_peaks(audio: np.ndarray, sr: int, samples_per_bucket: int = 441) -> dict:
    """Min/max per bucket (10 ms at 44.1 kHz) quantised to int8, for fast waveform drawing."""
    mono = to_mono(audio)
    n = len(mono) // samples_per_bucket
    if n == 0:
        return {"sample_rate": sr, "samples_per_bucket": samples_per_bucket, "min": [], "max": []}
    frames = mono[: n * samples_per_bucket].reshape(n, samples_per_bucket)
    mins = np.clip(np.round(frames.min(axis=1) * 127), -127, 127).astype(int)
    maxs = np.clip(np.round(frames.max(axis=1) * 127), -127, 127).astype(int)
    return {
        "sample_rate": sr, "samples_per_bucket": samples_per_bucket,
        "min": mins.tolist(), "max": maxs.tolist(),
    }


def pitch_spectrogram_png(audio: np.ndarray, sr: int, out: Path) -> dict:
    """Constant-Q magnitude with one bin per semitone (MIDI 21..108), as an 8-bit PNG.

    Rows map 1:1 to piano-roll pitches so the editor can draw it behind the notes.
    """
    import librosa
    from PIL import Image

    mono = librosa.resample(to_mono(audio), orig_sr=sr, target_sr=22050, res_type="soxr_hq")
    hop = 256
    fmin = librosa.midi_to_hz(21)
    C = np.abs(librosa.cqt(mono, sr=22050, hop_length=hop, fmin=fmin, n_bins=88, bins_per_octave=12))
    db = librosa.amplitude_to_db(C, ref=np.max(C) if C.size and np.max(C) > 0 else 1.0, top_db=70)
    img = np.clip((db + 70) / 70 * 255, 0, 255).astype(np.uint8)[::-1]  # high pitch on top
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG", optimize=True)
    out.write_bytes(buf.getvalue())
    return {"hop_seconds": hop / 22050, "midi_low": 21, "midi_high": 108, "frames": int(img.shape[1])}
