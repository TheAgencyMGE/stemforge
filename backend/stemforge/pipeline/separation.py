"""Source separation behind a replaceable adapter interface."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import get_settings

log = logging.getLogger(__name__)

ProgressFn = Callable[[float], None]  # fraction 0..1 of this separator's work; may raise JobCancelled


class Separator(Protocol):
    name: str
    sources: list[str]
    samplerate: int

    def separate(self, audio: np.ndarray, sr: int, on_progress: ProgressFn) -> dict[str, np.ndarray]: ...


def resolve_device() -> str:
    import torch

    pref = get_settings().device
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        raise RuntimeError("STEMFORGE_DEVICE=cuda but CUDA is not available")
    return "cpu"


_MODEL_CACHE: dict[str, object] = {}


def overlap_add(pieces: list[tuple[int, int, np.ndarray]], n: int, overlap: int) -> np.ndarray:
    """Crossfade overlapping chunks back onto one timeline.

    ``pieces`` are ``(start, end, array)`` with ``array.shape[-1] == end - start``. The fade is
    the same ramp the separator used before this helper existed: a chunk that touches t=0 or the
    end of the file is not faded on that edge, so a sample's time is never shifted.
    """
    if not pieces:
        raise ValueError("no pieces to overlap-add")
    out = np.zeros(pieces[0][2].shape[:-1] + (n,), dtype=np.float32)
    norm = np.zeros(n, dtype=np.float32)
    for lo, hi, res in pieces:
        length = hi - lo
        if res.shape[-1] != length:
            raise ValueError("piece length does not match its window")
        w = np.ones(length, dtype=np.float32)
        ramp = min(2 * overlap, length // 2)
        if lo > 0 and ramp > 0:
            w[:ramp] = np.linspace(0, 1, ramp, dtype=np.float32)
        if hi < n and ramp > 0:
            w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1, 0, ramp, dtype=np.float32))
        out[..., lo:hi] += res.astype(np.float32) * w
        norm[lo:hi] += w
    return out / np.maximum(norm, 1e-8)


class DemucsSeparator:
    """HTDemucs inference with outer chunking for bounded memory, progress and cancellation.

    Long inputs are processed in ``chunk`` second windows with overlapping, crossfaded borders.
    Demucs' own 7.8 s segmenting (with 25% overlap) runs inside each window. Normalisation uses
    whole-track statistics so every chunk sees the same gain.
    """

    def __init__(self, model_name: str):
        from ..model_registry import load_demucs, manifest

        self.name = f"demucs:{model_name}"
        self.model_name = model_name
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = load_demucs(model_name)
        self.model = _MODEL_CACHE[model_name]
        self.sources = list(self.model.sources)  # type: ignore[attr-defined]
        self.samplerate = int(self.model.samplerate)  # type: ignore[attr-defined]
        self.caveat = manifest()[model_name].get("caveat")

    def separate(self, audio: np.ndarray, sr: int, on_progress: ProgressFn) -> dict[str, np.ndarray]:
        import torch
        from demucs.apply import apply_model

        if sr != self.samplerate:
            raise ValueError(f"expected {self.samplerate} Hz input, got {sr}")
        settings = get_settings()
        device = resolve_device()
        n = audio.shape[1]
        chunk = int(settings.separation_chunk_seconds * sr)
        overlap = int(2.0 * sr)
        wav = torch.from_numpy(audio)
        ref = wav.mean(0)
        mean, std = float(ref.mean()), float(ref.std()) or 1.0
        wav = (wav - mean) / std

        starts = list(range(0, n, chunk)) or [0]
        segment = settings.demucs_segment
        t0 = time.monotonic()
        pieces: list[tuple[int, int, np.ndarray]] = []
        for i, s in enumerate(starts):
            lo, hi = max(0, s - overlap), min(n, s + chunk + overlap)
            piece = wav[:, lo:hi]
            res = self._run(apply_model, piece, device, segment)
            pieces.append((lo, hi, res))
            on_progress((i + 1) / len(starts))
            log.info("separation chunk done", extra={
                "model": self.model_name, "chunk": i + 1, "of": len(starts),
                "elapsed": round(time.monotonic() - t0, 2), "device": device,
            })
        # Demucs' own CLI adds the mixture mean back to every source, which puts N copies of the
        # mixture DC into the sum and a DC floor into "silent" stems. Split it once across sources.
        out = overlap_add(pieces, n, overlap) * std + mean / len(self.sources)
        return {src: out[k] for k, src in enumerate(self.sources)}

    def _run(self, apply_model, piece, device: str, segment: float | None) -> np.ndarray:
        import torch

        attempts = [(device, segment)]
        if device == "cuda":
            attempts += [("cuda", 4.0), ("cpu", segment)]  # OOM fallbacks: smaller segments, then CPU
        last: Exception | None = None
        for dev, seg in attempts:
            try:
                with torch.inference_mode():
                    kwargs = {"shifts": get_settings().demucs_shifts, "split": True, "overlap": 0.25,
                              "progress": False, "device": dev}
                    if seg:
                        kwargs["segment"] = seg
                    res = apply_model(self.model, piece[None], **kwargs)[0]
                return res.cpu().numpy().astype(np.float32)
            except torch.cuda.OutOfMemoryError as exc:
                last = exc
                log.warning("CUDA OOM during separation; retrying with fallback", extra={"next": dev})
                torch.cuda.empty_cache()
        raise RuntimeError(f"separation failed on all devices: {last}")


class AudioSeparatorAdapter:
    """Optional adapter for python-audio-separator (UVR / Roformer models). Not installed by default:

    the package downloads community weights whose licenses are mostly unstated. Enable with
    ``pip install audio-separator`` and ``STEMFORGE_SEPARATOR=audio_separator``, and set
    ``STEMFORGE_AUDIO_SEPARATOR_MODEL`` to a model filename you have reviewed.
    """

    def __init__(self, model_filename: str):
        try:
            from audio_separator.separator import Separator as _Sep  # type: ignore
        except ImportError as exc:
            raise RuntimeError("audio-separator is not installed") from exc
        self._cls = _Sep
        self.model_filename = model_filename
        self.name = f"audio_separator:{model_filename}"
        self.sources = ["vocals", "instrumental"]
        self.samplerate = 44100

    def separate(self, audio: np.ndarray, sr: int, on_progress: ProgressFn) -> dict[str, np.ndarray]:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "mix.wav"
            sf.write(str(src), audio.T, sr, subtype="FLOAT")
            sep = self._cls(output_dir=td, model_file_dir=str(get_settings().models_dir / "audio_separator"))
            sep.load_model(model_filename=self.model_filename)
            files = sep.separate(str(src))
            on_progress(1.0)
            result = {}
            for f in files:
                lower = Path(f).name.lower()
                key = "vocals" if "vocal" in lower else "instrumental"
                data, _ = sf.read(str(Path(td) / Path(f).name), dtype="float32", always_2d=True)
                result[key] = data.T
            return result


def gate_mixture_silence(mix: np.ndarray, stems: dict[str, np.ndarray], sr: int, floor_db: float = -90.0,
                         min_seconds: float = 0.05, fade_seconds: float = 0.005) -> dict[str, np.ndarray]:
    """Zero every stem where the *input mixture* is digitally silent.

    The sources must sum to the mixture, so where the mixture is below ``floor_db`` for at least
    ``min_seconds`` the only consistent estimate is silence; anything a separator emits there is a
    model artifact. Short fades avoid clicks. Musical content is never touched: the gate only acts
    where the mixture itself has nothing. Lengths and alignment are unchanged.
    """
    n = mix.shape[-1]
    hop = max(1, int(0.001 * sr))
    frames = int(np.ceil(n / hop))
    padded = np.zeros((mix.shape[0], frames * hop), dtype=np.float32)
    padded[:, :n] = np.abs(mix)
    peak = padded.reshape(mix.shape[0], frames, hop).max(axis=(0, 2))
    quiet = peak < 10 ** (floor_db / 20)
    keep = np.ones(frames, dtype=np.float32)
    min_frames = max(1, int(round(min_seconds * sr / hop)))
    i = 0
    while i < frames:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < frames and quiet[j]:
            j += 1
        if j - i >= min_frames:
            keep[i:j] = 0.0
        i = j
    if keep.all():
        return stems
    hard = np.repeat(keep, hop)[:n]
    fade = max(1, int(fade_seconds * sr))
    # Ramps sit inside the silent run, so audible samples always keep unity gain.
    smooth = np.convolve(hard, np.ones(fade, dtype=np.float32) / fade, mode="same") if fade > 1 else hard
    gain = np.where(hard > 0, 1.0, np.clip(smooth, 0.0, 1.0)).astype(np.float32)
    return {name: (data * gain).astype(np.float32) for name, data in stems.items()}


def residual_db(mix: np.ndarray, stems: dict[str, np.ndarray]) -> float:
    """Energy of mix - sum(stems) relative to the mix, in dB. Very negative = stems reconstruct the mix."""
    total = sum(stems.values())
    err = mix - total
    num = float(np.mean(err.astype(np.float64) ** 2))
    den = float(np.mean(mix.astype(np.float64) ** 2)) or 1e-12
    return round(10 * np.log10(max(num, 1e-12) / den), 2)
