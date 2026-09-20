"""Small, deterministic manifests for local reproducibility and release checks."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment_metadata():
    source = Path(__file__).parent
    project = source.parent.parent
    versions = {}
    for name in ["safeview-ml", "scikit-learn", "numpy", "scipy", "pandas", "joblib", "matplotlib", "gradio"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(project), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    digests = {str(p.relative_to(project)): sha256(p) for p in sorted(source.glob("*.py"))}
    if (project / "pyproject.toml").exists():
        digests["pyproject.toml"] = sha256(project / "pyproject.toml")
    return {
        "created_at": utc_now(), "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "versions": versions,
        "git_commit": commit, "git_dirty": dirty, "source_sha256": digests,
    }
