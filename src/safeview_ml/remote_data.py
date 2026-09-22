"""Read the author's ViHSD ZIP directly from GitHub, without extracting it.

The default keeps all corpus bytes in memory. An explicitly supplied cache
directory stores only the original ZIP, keyed by its immutable commit SHA.
No alternate dataset or mirror is used if the official source is unavailable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

from .data import DatasetBundle, bundle_from_frames

GITHUB_REPOSITORY = "sonlam1102/vihsd"
GITHUB_SOURCE = f"https://github.com/{GITHUB_REPOSITORY}"
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30
_SHA = re.compile(r"[0-9a-fA-F]{40}")
_SPLIT_FILES = {"train": "vihsd/train.csv", "validation": "vihsd/dev.csv", "test": "vihsd/test.csv"}


class DatasetAccessError(RuntimeError):
    """An official source request failed; callers should not substitute data."""


def _fetch_bytes(url: str, maximum: int) -> bytes:
    accept = "application/vnd.github+json" if url.startswith("https://api.github.com/") else "application/octet-stream"
    request = Request(url, headers={"User-Agent": "safeview-ml-vihsd-loader/1", "Accept": accept})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > maximum:
                raise ValueError("GitHub response exceeds the dataset loader's size limit.")
            payload = response.read(maximum + 1)
    except HTTPError as exc:
        raise DatasetAccessError(
            f"Official ViHSD GitHub request failed (HTTP {exc.code}). Check access, network and revision. "
            "For API rate limits, retry later or supply a known full commit SHA. No replacement dataset was used."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise DatasetAccessError("Cannot reach the official ViHSD GitHub source. Check network access and retry; no replacement dataset was used.") from exc
    if len(payload) > maximum:
        raise ValueError("GitHub response exceeds the dataset loader's size limit.")
    return payload


def resolve_github_revision(revision: str = "main") -> str:
    """Resolve branch/tag once; a full SHA avoids an unnecessary API request."""
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("A non-empty GitHub revision is required.")
    if _SHA.fullmatch(revision):
        return revision.lower()
    api_url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/commits/{quote(revision, safe='')}"
    try:
        info = json.loads(_fetch_bytes(api_url, 1024 * 1024))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DatasetAccessError("GitHub returned invalid revision metadata; no replacement dataset was used.") from exc
    sha = info.get("sha") if isinstance(info, dict) else None
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise DatasetAccessError("GitHub did not return a full immutable commit SHA.")
    return sha.lower()


def _read_archive(payload: bytes, revision: str, requested_revision: str) -> DatasetBundle:
    frames, files = {}, []
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(entries) > 100 or len(set(names)) != len(names):
                raise ValueError("ViHSD ZIP has too many or duplicate members.")
            if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name for name in names):
                raise ValueError("ViHSD ZIP has unsafe member paths.")
            if sum(entry.file_size for entry in entries) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("ViHSD ZIP exceeds the uncompressed size limit.")
            for split, name in _SPLIT_FILES.items():
                if name not in names:
                    raise ValueError(f"Official ViHSD ZIP is missing {name}; no replacement split was used.")
                entry = archive.getinfo(name)
                if entry.flag_bits & 1:
                    raise ValueError("Official ViHSD ZIP contains an encrypted split; obtain authorized access before retrying.")
                raw = archive.read(entry)
                frames[split] = pd.read_csv(io.BytesIO(raw), keep_default_na=False, dtype=str, encoding="utf-8-sig")
                files.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest()})
    except (BadZipFile, NotImplementedError, UnicodeDecodeError) as exc:
        raise ValueError("Official ViHSD archive is invalid or uses an unsupported ZIP/CSV format.") from exc
    bundle = bundle_from_frames(frames, source=GITHUB_SOURCE, revision=revision, files=files)
    bundle.manifest.update({
        "requested_revision": requested_revision,
        "source_type": "github",
        "source_url": f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{revision}/data/vihsd.zip",
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "provenance_note": "Author's GitHub archive at an immutable commit; dev mapped to validation, all original rows and split assignments retained. Publisher's research-use conditions still apply.",
    })
    return bundle


def load_github_dataset(revision: str = "main", cache_dir: str | Path | None = None) -> DatasetBundle:
    """Fetch original splits into memory; optionally reuse a user-chosen ZIP cache.

    Pass ``bundle.manifest['revision']`` on subsequent sessions to reuse exactly
    the same source commit. For Colab, an optional cache can live under
    ``/content/.cache/safeview/vihsd``; no project data directory is required.
    """
    resolved = resolve_github_revision(revision)
    url = f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{resolved}/data/vihsd.zip"
    cache = Path(cache_dir).expanduser() / f"{resolved}.zip" if cache_dir is not None else None
    if cache is not None and cache.is_file():
        if cache.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Cached ViHSD ZIP exceeds the size limit; remove the invalid cache file and retry.")
        payload = cache.read_bytes()
    else:
        payload = _fetch_bytes(url, MAX_ARCHIVE_BYTES)
    bundle = _read_archive(payload, resolved, revision)
    # Cache only after validating the archive and every split. Never extract CSVs.
    if cache is not None and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache.parent, prefix=".vihsd-", delete=False) as handle:
                staged = Path(handle.name)
                handle.write(payload)
            os.replace(staged, cache)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
    return bundle
