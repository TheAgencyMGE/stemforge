"""Verified model download and loading.

Only files listed in model_manifest.json are fetched, from a pinned commit, and each is
SHA-256 verified before it is moved into the model directory, and again before it is loaded.
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
    return get_settings().models_dir / "demucs" / name


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


def download(name: str, progress: bool = True) -> None:
    if name not in manifest():
        raise ModelError(f"unknown model {name!r}; known: {sorted(manifest())}")
    spec = manifest()[name]
    dest_dir = model_dir(name)
    dest_dir.mkdir(parents=True, exist_ok=True)
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
