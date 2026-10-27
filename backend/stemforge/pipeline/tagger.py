"""Audio tagging with PANNs Cnn14 (16 kHz), used as instrument-presence evidence.

Architecture after Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio
Pattern Recognition" (2020), code at github.com/qiuqiangkong/audioset_tagging_cnn (MIT,
Copyright (c) 2018-2020 Qiuqiang Kong). Weights: Zenodo record 3987831, CC BY 4.0. The
feature front end is reproduced with ``torch.stft`` and the checkpoint's own mel matrix, so
torchlibrosa is not needed.

The model answers "which of 527 AudioSet classes are audible in this 10 s clip". It does not
separate anything, and AudioSet labels are weak, clip-level labels: a probability is evidence
for presence, not a calibrated likelihood of a specific performer. Callers combine it with
energy-based activity before accepting a source.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SR = 16000
N_FFT = 512
HOP = 160
CLIP_SECONDS = 10.0
MODEL_NAME = "panns_cnn14_16k"


def _build():
    import torch
    from torch import nn
    from torch.nn import functional as F  # noqa: N812

    class ConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.bn2 = nn.BatchNorm2d(cout)

        def forward(self, x, pool):
            x = F.relu_(self.bn1(self.conv1(x)))
            x = F.relu_(self.bn2(self.conv2(x)))
            return F.avg_pool2d(x, kernel_size=pool)

    class Cnn14(nn.Module):
        def __init__(self, n_mels: int = 64, classes: int = 527):
            super().__init__()
            self.register_buffer("melW", torch.zeros(N_FFT // 2 + 1, n_mels))
            self.register_buffer("window", torch.hann_window(N_FFT, periodic=True))
            self.bn0 = nn.BatchNorm2d(n_mels)
            chans = [1, 64, 128, 256, 512, 1024, 2048]
            self.blocks = nn.ModuleList([ConvBlock(chans[i], chans[i + 1]) for i in range(6)])
            self.fc1 = nn.Linear(2048, 2048)
            self.fc_audioset = nn.Linear(2048, classes)

        def forward(self, wav):  # (batch, samples) at 16 kHz
            spec = torch.stft(wav, N_FFT, HOP, N_FFT, self.get_buffer("window"), center=True,
                              pad_mode="reflect", return_complex=True)
            power = (spec.real ** 2 + spec.imag ** 2).transpose(1, 2)  # (batch, time, freq)
            x = 10.0 * torch.log10(torch.clamp(power @ self.get_buffer("melW"), min=1e-10))[:, None]  # (b,1,t,mel)
            x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
            for i, block in enumerate(self.blocks):
                x = block(x, (2, 2) if i < 5 else (1, 1))
            x = torch.mean(x, dim=3)
            x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
            x = F.relu_(self.fc1(x))
            return torch.sigmoid(self.fc_audioset(x))

    return Cnn14


def convert_checkpoint(state: dict[str, Any]) -> dict[str, Any]:
    """Map the upstream state dict onto this module's names. Unused tensors (STFT conv kernels,
    the spec augmenter) are dropped; the mel matrix is kept as-is."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("spectrogram_extractor") or key.startswith("spec_augmenter"):
            continue
        if key == "logmel_extractor.melW":
            out["melW"] = value
        elif key.startswith("conv_block"):
            idx = int(key[len("conv_block")]) - 1
            out[f"blocks.{idx}.{key.split('.', 1)[1]}"] = value
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def load_model():
    import torch
    from safetensors.torch import load_file

    from ..model_registry import model_dir, verify

    verify(MODEL_NAME)
    tensors = load_file(str(model_dir(MODEL_NAME) / "cnn14_16k.safetensors"))
    model = _build()()
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    # The Hann window buffer is computed locally; nothing else may be missing.
    if [m for m in missing if m != "window"] or unexpected:
        raise RuntimeError(f"tagger checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return model


def available() -> bool:
    from ..model_registry import ModelError, verify

    try:
        verify(MODEL_NAME)
        return True
    except (ModelError, KeyError):
        return False


def tag_windows(audio: np.ndarray, sr: int, hop_seconds: float = 5.0, batch: int = 8) -> dict[str, np.ndarray]:
    """Clip-level AudioSet probabilities on 10 s windows every ``hop_seconds``.

    Returns ``{"probs": (windows, 527), "starts": (windows,) seconds}``. Inputs shorter than one
    window are zero-padded; the same zero padding is what the model saw in training.
    """
    import librosa
    import torch

    from .audio import to_mono

    y = librosa.resample(to_mono(audio).astype(np.float32), orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    win = int(CLIP_SECONDS * SR)
    hop = int(hop_seconds * SR)
    if y.size <= win:
        starts = [0]
    else:
        starts = list(range(0, y.size - win + 1, hop))
        if starts[-1] + win < y.size:
            starts.append(y.size - win)
    clips = np.zeros((len(starts), win), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = y[s:s + win]
        clips[i, : seg.size] = seg
    model = load_model()
    out = []
    with torch.inference_mode():
        for i in range(0, len(clips), batch):
            out.append(model(torch.from_numpy(clips[i:i + batch])).numpy())
    return {"probs": np.concatenate(out, axis=0), "starts": np.array(starts, dtype=np.float64) / SR}
