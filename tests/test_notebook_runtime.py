"""Portable notebook fixes must preserve unknown edits, dependencies and data."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/notebook_runtime.py"
spec = importlib.util.spec_from_file_location("notebook_runtime", MODULE_PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def runtime_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in runtime.RUNTIME_PATHS:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original: " + name + "\n")
    (source / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    for args in (["init", "--quiet"], ["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "baseline"]):
        subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)
    target = tmp_path / "runtime"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git"))
    changes = ["configs/experiments.yaml", "src/safeview_ml/data.py"]
    for name in changes:
        (source / name).write_text("reviewed fix: " + name + "\n")
    payload, digest = runtime.build_runtime_payload(source)
    return source, target, changes, payload, digest


def repack(payload, change):
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    change(contents)
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    raw = packed.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def save_notebook_payload(source, payload, digest):
    path = source / "notebooks/cs114_safeview.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": [
        f"_runtime_payload = {payload!r}\n", f"_runtime_digest = {digest!r}\n",
    ]}]}))


def test_runtime_payload_is_deterministic_and_repairs_old_files(runtime_copy):
    source, target, changes, payload, digest = runtime_copy
    assert runtime.build_runtime_payload(source) == (payload, digest)
    assert runtime.apply_runtime_payload(target, payload, digest) == changes
    assert (target / changes[0]).read_bytes() != (source / changes[0]).read_bytes()
    assert runtime.apply_runtime_payload(target, payload, digest, write=True) == changes
    assert runtime.apply_runtime_payload(target, payload, digest, write=True) == []
    for name in runtime.RUNTIME_PATHS:
        assert (target / name).read_bytes() == (source / name).read_bytes()


def test_standalone_source_can_repair_without_importing_repo(runtime_copy):
    _, target, changes, payload, digest = runtime_copy
    namespace = {}
    exec(runtime.RUNTIME_SYNC_SOURCE, namespace)
    assert namespace["apply_runtime_payload"](target, payload, digest, write=True) == changes


def test_payload_survives_commit_and_fresh_clone(runtime_copy, tmp_path):
    source, _, _, payload, digest = runtime_copy
    save_notebook_payload(source, payload, digest)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--quiet", "-m", "reviewed fixes and notebook"], check=True, capture_output=True)
    assert runtime.build_runtime_payload(source) == (payload, digest)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", str(source), str(clone)], check=True, capture_output=True)
    assert runtime.build_runtime_payload(clone) == (payload, digest)


def test_verified_saved_notebook_version_can_be_updated_without_hash_drift(runtime_copy):
    source, target, _, old_payload, old_digest = runtime_copy
    runtime.apply_runtime_payload(target, old_payload, old_digest, write=True)
    save_notebook_payload(source, old_payload, old_digest)
    name = "src/safeview_ml/data.py"
    (source / name).write_text("next reviewed fix\n")
    payload, digest = runtime.build_runtime_payload(source)
    assert runtime.apply_runtime_payload(target, payload, digest, write=True) == [name]
    assert (target / name).read_bytes() == (source / name).read_bytes()
    save_notebook_payload(source, payload, digest)
    assert runtime.build_runtime_payload(source) == (payload, digest)
    # A further notebook retains the original notebook's reviewed source too.
    (source / name).write_text("third reviewed fix\n")
    latest_payload, latest_digest = runtime.build_runtime_payload(source)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(old_payload))) as archive:
        (target / name).write_bytes(archive.read(name))
    assert runtime.apply_runtime_payload(target, latest_payload, latest_digest, write=True) == [name]


def test_saved_legacy_notebook_can_omit_newly_added_modules(runtime_copy):
    source, target, _, payload, _ = runtime_copy
    added = "src/safeview_ml/imbalance.py"

    def legacy(files):
        del files[added]
        manifest = json.loads(files["manifest.json"])
        manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != added]
        for entry in manifest["files"]:
            entry.pop("accepted_sha256", None)
        files["manifest.json"] = json.dumps(manifest).encode()

    old_payload, old_digest = repack(payload, legacy)
    save_notebook_payload(source, old_payload, old_digest)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(old_payload))) as archive:
        for name in runtime.RUNTIME_PATHS:
            if name != added:
                (target / name).write_bytes(archive.read(name))
    (target / added).unlink()
    name = "src/safeview_ml/data.py"
    (source / name).write_text("next reviewed fix\n")
    new_payload, new_digest = runtime.build_runtime_payload(source)
    assert set(runtime.apply_runtime_payload(target, new_payload, new_digest, write=True)) == {name, added}


@pytest.mark.parametrize("tamper", ["digest", "file", "path", "accepted"])
def test_tampered_saved_notebook_is_not_trusted(runtime_copy, tamper):
    source, _, _, payload, digest = runtime_copy

    def change(files):
        if tamper == "file":
            files["src/safeview_ml/data.py"] = b"tampered"
        elif tamper == "path":
            files["data/raw/train.csv"] = b"must not be trusted"
        elif tamper == "accepted":
            manifest = json.loads(files["manifest.json"])
            manifest["files"][0]["accepted_sha256"] = ["not-a-checksum"]
            files["manifest.json"] = json.dumps(manifest).encode()

    if tamper == "digest":
        digest = "0" * 64
    else:
        payload, digest = repack(payload, change)
    save_notebook_payload(source, payload, digest)
    with pytest.raises(ValueError):
        runtime.build_runtime_payload(source)


def test_saved_notebook_runtime_assignment_is_never_executed(runtime_copy, tmp_path):
    source, _, _, payload, digest = runtime_copy
    save_notebook_payload(source, payload, digest)
    marker = tmp_path / "executed"
    notebook = source / "notebooks/cs114_safeview.ipynb"
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": [
        f"_runtime_payload = __import__('pathlib').Path({str(marker)!r}).touch()\n",
        f"_runtime_digest = {digest!r}\n",
    ]}]}))
    with pytest.raises(ValueError, match="literal values"):
        runtime.build_runtime_payload(source)
    assert not marker.exists()


def test_unknown_edits_abort_all_writes(runtime_copy):
    _, target, changes, payload, digest = runtime_copy
    (target / "src/safeview_ml/training.py").write_text("user runtime changes\n")
    before = {name: (target / name).read_bytes() for name in runtime.RUNTIME_PATHS}
    with pytest.raises(ValueError, match="unrecognized changes"):
        runtime.apply_runtime_payload(target, payload, digest, write=True)
    assert {name: (target / name).read_bytes() for name in runtime.RUNTIME_PATHS} == before


def test_dependencies_and_user_data_are_preserved(runtime_copy):
    _, target, _, payload, digest = runtime_copy
    excluded = ["notebooks/cs114_safeview.ipynb", "data/raw/train.csv", "results/runs/user-run/progress.json"]
    for name in excluded:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("RUN_ID='mine'\nRUN_TRAINING=True\nuser content\n")
    before = {name: (target / name).read_bytes() for name in [*excluded, "pyproject.toml"]}
    runtime.apply_runtime_payload(target, payload, digest, write=True)
    assert {name: (target / name).read_bytes() for name in before} == before
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        assert set(archive.namelist()) == set(runtime.RUNTIME_PATHS) | {"manifest.json"}
        assert b"RUN_ID" not in b"".join(archive.read(name) for name in archive.namelist())


def test_changed_dependencies_abort_before_any_write(runtime_copy):
    _, target, changes, payload, digest = runtime_copy
    before = (target / changes[0]).read_bytes()
    (target / "pyproject.toml").write_text("user dependencies\n")
    with pytest.raises(ValueError, match="pyproject.toml differs"):
        runtime.apply_runtime_payload(target, payload, digest, write=True)
    assert (target / changes[0]).read_bytes() == before


def test_missing_known_source_is_supplied(runtime_copy):
    source, target, changes, payload, digest = runtime_copy
    missing = "src/safeview_ml/policy.py"
    (target / missing).unlink()
    changed = runtime.apply_runtime_payload(target, payload, digest, write=True)
    assert set(changed) == set(changes) | {missing}
    assert (target / missing).read_bytes() == (source / missing).read_bytes()


def test_tampered_payload_or_file_is_rejected(runtime_copy):
    _, target, _, payload, digest = runtime_copy
    with pytest.raises(ValueError, match="checksum mismatch"):
        runtime.apply_runtime_payload(target, payload, "0" * 64, write=True)
    altered, altered_digest = repack(payload, lambda files: files.update({"src/safeview_ml/data.py": b"tampered"}))
    with pytest.raises(ValueError, match="file checksum mismatch"):
        runtime.apply_runtime_payload(target, altered, altered_digest, write=True)


@pytest.mark.parametrize("path", ["../outside.py", "/tmp/outside.py", "data/raw/train.csv"])
def test_archive_traversal_and_unlisted_paths_rejected(runtime_copy, path):
    _, target, changes, payload, digest = runtime_copy
    before = (target / changes[0]).read_bytes()
    altered, altered_digest = repack(payload, lambda files: files.update({path: b"unlisted"}))
    with pytest.raises(ValueError, match="unexpected or duplicate paths"):
        runtime.apply_runtime_payload(target, altered, altered_digest, write=True)
    assert (target / changes[0]).read_bytes() == before


def test_symlink_target_is_rejected(runtime_copy, tmp_path):
    _, target, _, payload, digest = runtime_copy
    outside = tmp_path / "outside.py"
    outside.write_text("do not touch")
    name = target / "src/safeview_ml/data.py"
    name.unlink()
    name.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic-link"):
        runtime.apply_runtime_payload(target, payload, digest, write=True)
    assert outside.read_text() == "do not touch"


def test_manifest_cannot_override_fixed_allowlist(runtime_copy):
    _, target, _, payload, _ = runtime_copy
    def alter(files):
        manifest = json.loads(files["manifest.json"])
        manifest["files"][0]["path"] = "../outside.py"
        files["manifest.json"] = json.dumps(manifest).encode()
    altered, digest = repack(payload, alter)
    with pytest.raises(ValueError, match="unexpected or duplicate paths"):
        runtime.apply_runtime_payload(target, altered, digest, write=True)


def test_invalid_parent_aborts_before_any_write(runtime_copy):
    _, target, changes, payload, digest = runtime_copy
    before = (target / changes[0]).read_bytes()
    shutil.rmtree(target / "src")
    (target / "src").write_text("runtime user file")
    with pytest.raises(ValueError, match="parent is not a directory"):
        runtime.apply_runtime_payload(target, payload, digest, write=True)
    assert (target / changes[0]).read_bytes() == before
