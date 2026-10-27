"""Verified model download and loading.

Only files listed in model_manifest.json are fetched, from a pinned commit (or a fixed archival
record), and each is SHA-256 verified before it is moved into the model directory, and again
before it is loaded.

Runtime loading only ever reads ``.safetensors``. One upstream (PANNs on Zenodo) publishes a
PyTorch pickle only; for that entry the verified ``.pth`` is opened once with PyTorch's
restricted ``weights_only`` unpickler (tensors and primitive containers only, no code), converted
to safetensors, hashed, and deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger(__name__)
MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
HF_BASE = "https://huggingface.co"
ALLOWED_CLASS_PREFIX = "demucs."


class ModelError(Exception):
    pass


@lru_cache
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())["models"]


def model_dir(name: str) -> Path:
    return get_settings().models_dir / manifest()[name].get("dir", "demucs") / name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(name: str) -> dict[str, Any]:
    spec = manifest()[name]
    present = [(model_dir(name) / f["name"]).is_file() for f in spec["files"]]
    return {
        "name": name, "installed": all(present), "sources": spec["sources"],
        "size_bytes": sum(f["size"] for f in spec["files"]),
        "license": spec["license"], "caveat": spec.get("caveat"),
    }


def download(name: str, progress: bool = True, source_file: Path | None = None) -> None:
    if name not in manifest():
        raise ModelError(f"unknown model {name!r}; known: {sorted(manifest())}")
    spec = manifest()[name]
    dest_dir = model_dir(name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if "convert" in spec:
        _download_and_convert(name, spec, dest_dir, progress, source_file)
        log.info("model installed", extra={"model": name})
        return
    for f in spec["files"]:
        dest = dest_dir / f["name"]
        if dest.is_file() and _sha256(dest) == f["sha256"]:
            log.info("model file already verified", extra={"model": name, "file": f["name"]})
            continue
        if not f["name"].endswith(".safetensors"):
            raise ModelError("refusing to download non-safetensors model file")
        url = f"{HF_BASE}/{spec['repo']}/resolve/{spec['revision']}/{f['name']}"
        fd, tmp_name = tempfile.mkstemp(dir=dest_dir, suffix=".part")
        h = hashlib.sha256()
        done = 0
        try:
            with os.fdopen(fd, "wb") as out, httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_bytes(1 << 20):
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if done > f["size"] * 1.01 + 1024:
                        raise ModelError("download larger than manifest size; aborting")
                    if progress:
                        print(f"\r{name}/{f['name']}: {done / 1e6:6.1f} / {f['size'] / 1e6:.1f} MB", end="", flush=True)
            if progress:
                print()
            if h.hexdigest() != f["sha256"]:
                raise ModelError(f"SHA-256 mismatch for {f['name']}: got {h.hexdigest()}")
            os.replace(tmp_name, dest)
        finally:
            Path(tmp_name).unlink(missing_ok=True)
    log.info("model installed", extra={"model": name})


def _fetch(url: str, dest_dir: Path, size: int, label: str, progress: bool) -> tuple[Path, str, str]:
    """Stream ``url`` to a temp file under ``dest_dir``; return (path, sha256, md5)."""
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir, suffix=".part")
    sha, md5 = hashlib.sha256(), hashlib.md5()  # noqa: S324 - md5 only to match the publisher's checksum
    done = 0
    try:
        with os.fdopen(fd, "wb") as out, httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes(1 << 20):
                out.write(chunk)
                sha.update(chunk)
                md5.update(chunk)
                done += len(chunk)
                if done > size * 1.01 + 1024:
                    raise ModelError("download larger than manifest size; aborting")
                if progress:
                    print(f"\r{label}: {done / 1e6:6.1f} / {size / 1e6:.1f} MB", end="", flush=True)
        if progress:
            print()
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return Path(tmp_name), sha.hexdigest(), md5.hexdigest()


def _download_and_convert(name: str, spec: dict[str, Any], dest_dir: Path, progress: bool,
                          source_file: Path | None = None) -> None:
    out_file = spec["files"][0]
    dest = dest_dir / out_file["name"]
    if dest.is_file() and _sha256(dest) == out_file["sha256"]:
        return
    src = spec["source_file"]
    if source_file is not None:
        # A copy fetched earlier (or offline). It goes through exactly the same checks.
        fd, tmp_name = tempfile.mkstemp(dir=dest_dir, suffix=".part")
        os.close(fd)
        tmp = Path(tmp_name)
        import shutil

        shutil.copyfile(source_file, tmp)
        md5_h = hashlib.md5()  # noqa: S324 - matches the publisher's checksum
        with tmp.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5_h.update(chunk)
        sha, md5 = _sha256(tmp), md5_h.hexdigest()
    else:
        tmp, sha, md5 = _fetch(spec["url"], dest_dir, src["size"], f"{name}/{src['name']}", progress)
    try:
        if sha != src["sha256"] or md5 != src["md5"]:
            raise ModelError(f"checksum mismatch for {src['name']}: sha256={sha} md5={md5}")
        if spec["convert"] != "panns_cnn14":
            raise ModelError(f"unknown conversion {spec['convert']!r}")
        import numpy as np
        import torch
        from numpy._core.multiarray import _reconstruct
        from safetensors.torch import save_file

        from .pipeline.tagger import convert_checkpoint

        # The checkpoint also stores the training sampler state as NumPy arrays. Allow exactly the
        # NumPy array/dtype reconstructors in the restricted unpickler; that state is then dropped.
        allowed: list[Any] = [(_reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype,
                   *(type(np.dtype(t)) for t in ("int64", "int32", "float64", "float32", "bool", "uint8"))]
        with torch.serialization.safe_globals(allowed):
            ckpt = torch.load(str(tmp), map_location="cpu", weights_only=True)
        state = {k: v for k, v in convert_checkpoint(ckpt.get("model", ckpt)).items() if torch.is_tensor(v)}
        part = dest.with_suffix(".safetensors.part")
        save_file({k: v.contiguous() for k, v in state.items()}, str(part),
                  metadata={"source_sha256": sha})
        got = _sha256(part)
        if got != out_file["sha256"]:
            part.unlink(missing_ok=True)
            raise ModelError(f"converted file hash mismatch: {got}")
        os.replace(part, dest)
    finally:
        tmp.unlink(missing_ok=True)


def verify(name: str) -> None:
    spec = manifest()[name]
    for f in spec["files"]:
        p = model_dir(name) / f["name"]
        if not p.is_file():
            raise ModelError(
                f"model {name} is not installed. Run: stemforge-models download {name}"
            )
        if _sha256(p) != f["sha256"]:
            raise ModelError(f"model file {p.name} failed SHA-256 verification; re-download it")


def load_demucs(name: str):
    """Build a demucs BagOfModels from verified safetensors, restricting classes to demucs.*."""
    import importlib

    from demucs.apply import BagOfModels
    from demucs.hf import _decode_json, _unflatten_state
    from demucs.states import load_model
    from safetensors import safe_open

    verify(name)
    spec = manifest()[name]
    models = []
    for sig in spec["bag"]["models"]:
        path = model_dir(name) / f"{sig}.safetensors"
        with safe_open(str(path), framework="pt") as file:
            metadata = file.metadata() or {}
            tensors = {key: file.get_tensor(key) for key in file.keys()}
        klass_path = metadata.get("klass", "")
        if not klass_path.startswith(ALLOWED_CLASS_PREFIX):
            raise ModelError(f"refusing to instantiate non-demucs class {klass_path!r}")
        state = _unflatten_state(tensors, json.loads(metadata["structure"])) if "structure" in metadata else tensors
        module, cls = klass_path.rsplit(".", 1)
        klass = getattr(importlib.import_module(module), cls)
        args = _decode_json(json.loads(metadata["args"]))
        kwargs = _decode_json(json.loads(metadata["kwargs"]))
        models.append(load_model({"klass": klass, "args": args, "kwargs": kwargs, "state": state}))
    bag = BagOfModels(models, spec["bag"].get("weights"), None)
    bag.eval()
    return bag
