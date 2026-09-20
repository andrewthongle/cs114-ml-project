#!/usr/bin/env python3
"""Fetch authorized ViHSD files at a resolved, immutable HF revision.

The dataset is gated. Authenticate using ``hf auth login`` or HF_TOKEN and
accept the publisher's access terms on Hugging Face before running this script.
Credentials are neither stored in the dataset directory nor printed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from safeview_ml.data import load_local_dataset, save_manifest

DATASET_ID = "uitnlp/vihsd"


def download_dataset(output_dir: str | Path, revision: str = "main") -> dict:
    """Stage a complete download and validate official splits before publishing."""
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
    parser.add_argument("--output", default="data/raw/vihsd", help="Absent or empty destination directory")
    parser.add_argument("--revision", default="main", help="HF branch/tag/SHA; resolved to immutable SHA before downloading")
    args = parser.parse_args()
    try:
        manifest = download_dataset(args.output, args.revision)
    except HfHubHTTPError:
        parser.exit(1, "ViHSD download was not authorized or the Hub request failed. Verify network access, local HF authentication, dataset access approval and requested revision. No replacement data was used.\n")
    except (PermissionError, FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({"output": str(Path(args.output).resolve()), "revision": manifest["revision"], "fingerprint": manifest["fingerprint"], "split_counts": manifest["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
