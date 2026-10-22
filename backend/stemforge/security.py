"""Upload sniffing, filename sanitisation, signed URLs and rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
import unicodedata
from collections import OrderedDict

# Container detection from magic bytes. Extensions are never trusted.
AUDIO_FORMATS = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "aiff": "audio/aiff",
    "m4a": "audio/mp4",
    "webm": "audio/webm",
    "caf": "audio/x-caf",
}


def sniff_audio_format(head: bytes) -> str | None:
    """Return a format id for a recognised audio container header, else None."""
    if len(head) < 12:
        return None
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"caff":
        return "caf"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B ", b"mp42", b"isom", b"mp41", b"iso2", b"dash", b"MSNV", b"M4V "):
            return "m4a"
        return None
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[:3] == b"ID3":
        return "mp3"
    # Raw MPEG audio frame sync: 11 set bits, layer III/II, not a reserved version.
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0 and (head[1] & 0x06) != 0 and (head[1] & 0x18) != 0x08:
        return "mp3"
    # ADTS AAC
    if head[0] == 0xFF and (head[1] & 0xF6) == 0xF0:
        return "m4a"
    return None


_UNSAFE = re.compile(r"[^A-Za-z0-9 ._()\-\[\]]+")


def safe_filename(name: str, default: str = "audio") -> str:
    """Strip directories, control characters and odd unicode; keep something human readable."""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"\s+", " ", _UNSAFE.sub("_", name)).strip(" ._")
    if not name or name in {".", ".."}:
        name = default
    return name[:120]


def display_name_from_filename(name: str) -> str:
    stem = safe_filename(name).rsplit(".", 1)[0]
    return (stem.replace("_", " ").strip() or "Untitled")[:200]


def sign(secret: str, artifact_id: str, expires: int) -> str:
    msg = f"{artifact_id}:{expires}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify_signature(secret: str, artifact_id: str, expires: int, signature: str) -> bool:
    if expires < int(time.time()):
        return False
    return hmac.compare_digest(sign(secret, artifact_id, expires), signature)


class RateLimiter:
    """Token bucket per client key. In-process; put a shared limiter in front for multi-instance."""

    def __init__(self, per_minute: int, max_keys: int = 10_000):
        self.rate = per_minute / 60.0
        self.capacity = float(max(per_minute, 1))
        self.max_keys = max_keys
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        if self.rate <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.pop(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            ok = tokens >= cost
            if ok:
                tokens -= cost
            self._buckets[key] = (tokens, now)
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
            return ok
