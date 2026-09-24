"""Carry reviewed source fixes with a notebook, without changing its settings.

The payload contains only the listed Python modules and experiment defaults.
Runtime edits are accepted only when they match the build's Git baseline,
the carried version, or an integrity-checked prior notebook version;
dependencies and research outputs are never overwritten.
"""
import ast
import base64
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import zipfile


RUNTIME_PATHS = (
    "configs/experiments.yaml",
    "configs/traditional.yaml",
    "scripts/publish_hf.py",
    "scripts/templates/hf_space/app.py",
    "scripts/templates/hf_space/README.md",
    "src/safeview_ml/__init__.py",
    "src/safeview_ml/cli.py",
    "src/safeview_ml/data.py",
    "src/safeview_ml/evaluation.py",
    "src/safeview_ml/features.py",
    "src/safeview_ml/imbalance.py",
    "src/safeview_ml/inference.py",
    "src/safeview_ml/models.py",
    "src/safeview_ml/policy.py",
    "src/safeview_ml/preprocessing.py",
    "src/safeview_ml/provenance.py",
    "src/safeview_ml/remote_data.py",
    "src/safeview_ml/training.py",
    "src/safeview_ml/transformer_model.py",
)


def _saved_notebook_payload(root):
    """Read prior payload data as literals; never execute a saved notebook."""
    notebook_path = root / "notebooks/cs114_safeview.ipynb"
    if not notebook_path.exists():
        return None
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    values = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = source if isinstance(source, str) else "".join(source)
        if "_runtime_payload" not in source and "_runtime_digest" not in source:
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("_runtime_payload", "_runtime_digest"):
                    if target.id in values:
                        raise ValueError("Saved notebook has duplicate runtime payload assignments")
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("Saved notebook runtime payload must use literal values") from exc
    if not values:
        return None
    if set(values) != {"_runtime_payload", "_runtime_digest"}:
        raise ValueError("Saved notebook has an incomplete runtime payload")
    _, manifest, records = _decode_runtime_payload(
        values["_runtime_payload"], values["_runtime_digest"], allow_subset=True,
    )
    return values["_runtime_payload"], values["_runtime_digest"], manifest, records


def _saved_notebook_hashes(root):
    saved = _saved_notebook_payload(root)
    return _record_hashes(saved[3]) if saved else {}


def _record_hashes(records):
    return {
        name: {entry["sha256"], *entry.get("accepted_sha256", []), *(
            [entry["previous_sha256"]] if entry.get("previous_sha256") else []
        )}
        for name, entry in records.items()
    }


def build_runtime_payload(root):
    """Return a deterministic base64 ZIP and its SHA-256 from trusted sources."""
    root = Path(root).resolve()
    saved = _saved_notebook_payload(root)
    if saved:
        payload, digest, manifest, records = saved
        # A commit or fresh clone must not change the notebook merely because
        # HEAD now contains the carried sources. Retain the validated payload
        # (and accepted older versions) until actual source/dependencies change.
        if (set(records) == set(RUNTIME_PATHS)
                and manifest["pyproject_sha256"] == hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest()
                and all(records[name]["sha256"] == hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in RUNTIME_PATHS)):
            return payload, digest
    prior_hashes = _record_hashes(saved[3]) if saved else {}
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    contents, entries = {}, []
    for name in RUNTIME_PATHS:
        content = (root / name).read_bytes()
        baseline = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{name}"],
            capture_output=True, check=False,
        )
        contents[name] = content
        current_hash = hashlib.sha256(content).hexdigest()
        baseline_hash = hashlib.sha256(baseline.stdout).hexdigest() if baseline.returncode == 0 else None
        entries.append({
            "path": name,
            "sha256": current_hash,
            "previous_sha256": baseline_hash,
            # Exclude current/HEAD so rebuilding the saved notebook does not
            # accumulate self-referential history or change its payload hash.
            "accepted_sha256": sorted(prior_hashes.get(name, set()) - {current_hash, baseline_hash}),
        })
    manifest = {
        "schema_version": 1,
        "pyproject_sha256": hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest(),
        "files": entries,
    }
    contents["manifest.json"] = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(contents):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, contents[name])
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), hashlib.sha256(payload).hexdigest()


def _decode_runtime_payload(payload, digest, *, allow_subset=False):
    """Validate carried bytes against the fixed source/config allowlist."""
    import base64
    import binascii
    import hashlib
    import io
    import json
    import re
    import zipfile

    def checksum(content):
        return hashlib.sha256(content).hexdigest()

    def valid_hash(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    if not isinstance(payload, str) or len(payload) > 4 * 1024 * 1024 or not valid_hash(digest):
        raise ValueError("Invalid notebook runtime payload or digest")
    try:
        packed = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid notebook runtime payload encoding") from exc
    if checksum(packed) != digest:
        raise ValueError("Notebook runtime payload checksum mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(packed)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            allowed = set(RUNTIME_PATHS) | {"manifest.json"}
            if (len(names) != len(set(names)) or "manifest.json" not in names
                    or not set(names) <= allowed
                    or (not allow_subset and set(names) != allowed)):
                raise ValueError("Notebook runtime archive has unexpected or duplicate paths")
            if any(member.file_size > 1024 * 1024 for member in members):
                raise ValueError("Notebook runtime archive file exceeds its size limit")
            if sum(member.file_size for member in members) > 4 * 1024 * 1024:
                raise ValueError("Notebook runtime archive exceeds its size limit")
            contents = {name: archive.read(name) for name in names}
        manifest = json.loads(contents.pop("manifest.json"))
    except (zipfile.BadZipFile, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid notebook runtime archive or manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported notebook runtime manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries or len(entries) != len(contents):
        raise ValueError("Notebook runtime manifest has invalid file entries")
    if not valid_hash(manifest.get("pyproject_sha256")):
        raise ValueError("Notebook runtime manifest has invalid dependency checksum")
    records = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Notebook runtime manifest has invalid file entries")
        name = entry.get("path")
        if not isinstance(name, str) or name not in contents or name not in RUNTIME_PATHS or name in records:
            raise ValueError("Notebook runtime manifest has unexpected or duplicate paths")
        current, previous = entry.get("sha256"), entry.get("previous_sha256")
        if not valid_hash(current) or (previous is not None and not valid_hash(previous)):
            raise ValueError("Notebook runtime manifest has invalid file checksums")
        accepted = entry.get("accepted_sha256", [])
        if not isinstance(accepted, list) or any(not valid_hash(value) for value in accepted):
            raise ValueError("Notebook runtime manifest has invalid accepted checksums")
        if checksum(contents[name]) != current:
            raise ValueError(f"Notebook runtime file checksum mismatch: {name}")
        records[name] = entry
    return contents, manifest, records


def apply_runtime_payload(root, payload, digest, *, write=False):
    """Validate every file, then optionally replace known older files atomically.

    Return the paths that differ from the payload. Unknown local edits abort
    the entire operation before any file is changed. No modules are imported
    and no packages are installed by this helper.
    """
    import hashlib
    import os
    from pathlib import Path
    import tempfile

    def checksum(content):
        return hashlib.sha256(content).hexdigest()

    contents, manifest, records = _decode_runtime_payload(payload, digest)

    root = Path(root).resolve()

    def safe_target(name):
        target = root / name
        for path in (target, *target.parents):
            if path == root:
                break
            if path.is_symlink():
                raise ValueError(f"Notebook runtime refuses a symbolic-link path: {name}")
            if path != target and path.exists() and not path.is_dir():
                raise ValueError(f"Notebook runtime parent is not a directory: {name}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Notebook runtime target is not a file: {name}")
        return target

    prerequisite = safe_target("pyproject.toml")
    required_hash = manifest.get("pyproject_sha256")
    if not prerequisite.is_file() or checksum(prerequisite.read_bytes()) != required_hash:
        raise ValueError("Runtime pyproject.toml differs from this notebook; synchronize the repository and dependencies first")

    changed, targets = [], {}
    for name in RUNTIME_PATHS:
        target = safe_target(name)
        targets[name] = target
        actual = checksum(target.read_bytes()) if target.exists() else None
        record = records[name]
        if actual == record["sha256"]:
            continue
        known_prior = {record.get("previous_sha256"), *record.get("accepted_sha256", [])}
        if actual is not None and actual not in known_prior:
            raise ValueError(f"Runtime file has unrecognized changes; preserve and review it before syncing: {name}")
        changed.append(name)
    if not write:
        return changed
    for name in changed:
        target = targets[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".notebook-sync-", delete=False) as handle:
                staged = Path(handle.name)
                handle.write(contents[name])
            os.chmod(staged, target.stat().st_mode & 0o777 if target.exists() else 0o644)
            os.replace(staged, target)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
    return changed


# This standard-library-only source can be embedded in the notebook, allowing
# an older checkout to be repaired before safeview_ml is imported.
RUNTIME_SYNC_SOURCE = (
    "RUNTIME_PATHS = " + repr(RUNTIME_PATHS) + "\n\n"
    + inspect.getsource(_decode_runtime_payload) + "\n\n"
    + inspect.getsource(apply_runtime_payload)
)
