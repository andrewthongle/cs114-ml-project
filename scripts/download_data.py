#!/usr/bin/env python3
"""Validate the author's GitHub ViHSD dataset and save metadata by default.

Corpus bytes stay in memory unless --cache-dir is explicitly supplied. For an
authorized Hugging Face export, use --source huggingface --output DIRECTORY;
only that optional path requires Hub authentication and dataset access.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

try:
    from huggingface_hub import HfApi, get_token, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
    HUB_HTTP_ERRORS = (HfHubHTTPError,)
except ImportError:  # The default GitHub loader has no Hub dependency.
    HfApi = get_token = snapshot_download = None
    HUB_HTTP_ERRORS = ()

from safeview_ml.data import load_local_dataset, save_manifest
from safeview_ml.remote_data import load_github_dataset

DATASET_ID = "uitnlp/vihsd"


def download_dataset(output_dir: str | Path, revision: str = "main") -> dict:
    """Optional authenticated HF export; retained for existing local workflows."""
    if HfApi is None:
        raise RuntimeError("The optional Hugging Face export requires `pip install -e '.[hub]'`.")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Destination must be absent or empty: {destination}")
    token = get_token()
    if not token:
        raise PermissionError(
            "ViHSD access requires an authenticated Hugging Face account. Accept the dataset terms at "
            "https://huggingface.co/datasets/uitnlp/vihsd, then run `hf auth login` or set HF_TOKEN locally."
        )
    info = HfApi(token=token).dataset_info(DATASET_ID, revision=revision)
    if not info.sha:
        raise RuntimeError("Hugging Face did not return an immutable dataset revision.")
    snapshot = Path(snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=info.sha,
        token=token,
        allow_patterns=["*.csv", "*.parquet", "*.jsonl", "*.txt", "README.md"],
    ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".vihsd-download-", dir=destination.parent))
    try:
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and (path.suffix in {".csv", ".parquet", ".jsonl", ".txt"} or path.name == "README.md"):
                relative = path.relative_to(snapshot)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        metadata = {
            "dataset": "ViHSD",
            "source": f"https://huggingface.co/datasets/{DATASET_ID}",
            "revision": info.sha,
            "requested_revision": revision,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "synthetic": False,
            "access": "Downloaded through authenticated Hugging Face Hub API; publisher's terms still apply.",
        }
        (stage / "source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        bundle = load_local_dataset(stage)
        save_manifest(bundle, stage / "manifest.json")
        # Atomic rename on the same filesystem; refuses a nonempty destination.
        stage.rename(destination)
        return bundle.manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["github", "huggingface"], default="github")
    parser.add_argument("--manifest", default="data/manifest.json", help="Metadata-only JSON output for GitHub loading")
    parser.add_argument("--cache-dir", help="Optional ZIP cache outside the repository; default keeps corpus in memory")
    parser.add_argument("--output", help="Absent or empty destination directory, only for --source huggingface")
    parser.add_argument("--revision", default="main", help="Source branch/tag/SHA; resolved to immutable SHA before downloading")
    args = parser.parse_args()
    try:
        if args.source == "github":
            if args.output:
                raise ValueError("GitHub loading does not extract dataset files. Use --cache-dir for an optional ZIP cache, or --source huggingface for --output.")
            bundle = load_github_dataset(args.revision, cache_dir=args.cache_dir)
            save_manifest(bundle, args.manifest)
            manifest = bundle.manifest
            output = {"manifest": str(Path(args.manifest).resolve())}
        else:
            if not args.output or args.cache_dir:
                raise ValueError("--source huggingface requires --output and does not use --cache-dir.")
            manifest = download_dataset(args.output, args.revision)
            output = {"output": str(Path(args.output).resolve())}
    except HUB_HTTP_ERRORS:
        parser.exit(1, "ViHSD download was not authorized or the Hub request failed. Verify network access, local HF authentication, dataset access approval and requested revision. No replacement data was used.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({**output, "revision": manifest["revision"], "fingerprint": manifest["fingerprint"], "split_counts": manifest["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
