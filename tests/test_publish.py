"""Local packaging protocol tests; these bytes are not real model releases.

Every artifact is invented under tmp_path. Accepted-provenance fields exercise
the packaging protocol only; no smoke artifacts are reused or relabeled, no
joblib is deserialized, and constructing an HF API client fails every test.
"""

import importlib.util
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
import zipfile

import huggingface_hub
import pytest


SPEC = importlib.util.spec_from_file_location(
    "publish_hf_test_module", Path(__file__).resolve().parents[1] / "scripts/publish_hf.py"
)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


@pytest.fixture
def package_fixture(tmp_path, monkeypatch):
    project = tmp_path / "invented-project"
    source = project / "src/safeview_ml"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# Invented packaging test fixture, not a trained model.\n")
    (project / "pyproject.toml").write_text('[project]\nname = "invented-fixture"\nversion = "0.0.0"\n')
    space_source = project / "deploy/hf_space"
    space_source.mkdir(parents=True)
    (space_source / "app.py").write_text("# Invented serving fixture; no application or network code.\n")
    (space_source / "README.md").write_text('---\nsdk: gradio\npython_version: "3.13"\n---\n')
    monkeypatch.setattr(publisher, "PROJECT", project)
    release = tmp_path / "invented-release"
    evaluation = tmp_path / "invented-evaluation"
    release.mkdir()
    (evaluation / "fixture_family").mkdir(parents=True)
    (release / "pipeline.joblib").write_bytes(b"INVENTED FIXTURE: not a loadable joblib model")
    publisher.write_json(release / "decision_policy.json", {"fixture": True, "threshold": 0.5})
    hashes = {name: publisher.file_sha256(release / name) for name in ("pipeline.joblib", "decision_policy.json")}
    sources = {str(path.relative_to(project)): publisher.file_sha256(path)
               for path in [source / "__init__.py", project / "pyproject.toml"]}
    metadata = {
        "invented_protocol_fixture": True,
        "synthetic": False,
        "model_revision": "invented-revision",
        "dataset_fingerprint": "invented-fingerprint",
        "sha256": hashes,
        "source_sha256": sources,
        "python": platform.python_version(),
        "versions": {},
    }
    final = {
        "invented_protocol_fixture": True,
        "synthetic": False,
        "status": "evaluated",
        "model_revision": "invented-revision",
        "dataset_fingerprint": "invented-fingerprint",
        "artifact_sha256": hashes,
    }
    publisher.write_json(release / "metadata.json", metadata)
    publisher.write_json(evaluation / "metadata.json", final)
    (evaluation / "comparison.csv").write_text("family,invented_fixture\nfixture_family,true\n")
    publisher.write_json(evaluation / "fixture_family/test_metrics.json", {"invented_protocol_fixture": True})
    # These unapproved files must never enter a prepared upload folder.
    private_marker = "PRIVATE_DATA_MUST_NOT_BE_BUNDLED"
    (release / "raw_comments.csv").write_text(private_marker)
    (evaluation / "predictions.csv").write_text(private_marker)
    calls = []

    def fake_predictor(path):
        calls.append(Path(path))
        return SimpleNamespace(model_revision="invented-revision", policy_version="fixture-1",
                               metadata=publisher.read_json(Path(path) / "metadata.json"))

    def fake_build(command, *, cwd, check):
        assert "--wheel" in command and check and cwd == project
        wheel_dir = Path(command[command.index("--outdir") + 1])
        wheel_dir.mkdir()
        with zipfile.ZipFile(wheel_dir / "invented_fixture-0.0.0-py3-none-any.whl", "w") as archive:
            archive.write(source / "__init__.py", "safeview_ml/__init__.py")

    def forbid_hf(*args, **kwargs):
        pytest.fail("HF API clients are forbidden in local packaging tests")

    monkeypatch.setattr(publisher, "ReleasePredictor", fake_predictor)
    monkeypatch.setattr(publisher, "pinned_dependencies", lambda: ["invented-dependency==0.0.0"])
    monkeypatch.setattr(publisher.subprocess, "run", fake_build)
    monkeypatch.setattr(huggingface_hub, "HfApi", forbid_hf)
    return SimpleNamespace(release=release, evaluation=evaluation, metadata=metadata, final=final,
                           output=tmp_path / "prepared-fixture", calls=calls, private_marker=private_marker)


def test_rejects_synthetic_and_unknown_provenance_before_deserialization(package_fixture):
    fixture = package_fixture
    for flag in (True, None):
        publisher.write_json(fixture.release / "metadata.json", {**fixture.metadata, "synthetic": flag})
        with pytest.raises(ValueError, match="synthetic/unknown"):
            publisher.verify_release(fixture.release, fixture.evaluation)
    publisher.write_json(fixture.release / "metadata.json", fixture.metadata)
    publisher.write_json(fixture.evaluation / "metadata.json", {**fixture.final, "synthetic": True})
    with pytest.raises(ValueError, match="synthetic/unknown"):
        publisher.verify_release(fixture.release, fixture.evaluation)
    assert fixture.calls == []


def test_rejects_unfinished_final_evaluation(package_fixture):
    fixture = package_fixture
    publisher.write_json(fixture.evaluation / "metadata.json", {**fixture.final, "status": "evaluating"})
    with pytest.raises(ValueError, match="Complete the frozen final test"):
        publisher.verify_release(fixture.release, fixture.evaluation)
    assert fixture.calls == []


def test_rejects_wrong_model_revision_and_dataset_fingerprint(package_fixture):
    fixture = package_fixture
    for key, message in (("model_revision", "model revision"), ("dataset_fingerprint", "fingerprint")):
        publisher.write_json(fixture.evaluation / "metadata.json", {**fixture.final, key: "different"})
        with pytest.raises(ValueError, match=message):
            publisher.verify_release(fixture.release, fixture.evaluation)
    assert fixture.calls == []


def test_rejects_artifact_not_matching_final_evaluation(package_fixture):
    fixture = package_fixture
    publisher.write_json(fixture.evaluation / "metadata.json", {
        **fixture.final, "artifact_sha256": {**fixture.final["artifact_sha256"], "pipeline.joblib": "0" * 64}
    })
    with pytest.raises(ValueError, match="differs from evaluated artifact"):
        publisher.verify_release(fixture.release, fixture.evaluation)
    assert fixture.calls == []


def test_rejects_source_snapshot_drift(package_fixture):
    fixture = package_fixture
    publisher.write_json(fixture.release / "metadata.json", {**fixture.metadata, "source_sha256": {"wrong.py": "0" * 64}})
    with pytest.raises(ValueError, match="Training source differs"):
        publisher.verify_release(fixture.release, fixture.evaluation)
    assert fixture.calls == []


def test_rejects_python_and_dependency_runtime_drift(package_fixture, monkeypatch):
    fixture = package_fixture
    publisher.write_json(fixture.release / "metadata.json", {**fixture.metadata, "python": "0.0.0"})
    with pytest.raises(ValueError, match="Python version"):
        publisher.verify_release(fixture.release, fixture.evaluation)
    publisher.write_json(fixture.release / "metadata.json", {**fixture.metadata, "versions": {"invented-runtime": "1.0"}})
    monkeypatch.setattr(publisher.importlib.metadata, "version", lambda name: "2.0")
    with pytest.raises(ValueError, match="differs from the training environment"):
        publisher.verify_release(fixture.release, fixture.evaluation)
    assert fixture.calls == []


def test_local_bundle_is_complete_immutable_and_rejects_added_upload_data(package_fixture):
    fixture = package_fixture
    output = publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output)
    assert output == fixture.output
    assert fixture.calls == [fixture.release]
    manifest = publisher.verify_prepared_bundle(output)
    assert manifest["publish_status"] == "prepared_locally"
    assert (output / "model/evaluation/fixture_family/test_metrics.json").exists()
    assert (output / "space/app.py").exists()
    assert not list(output.parent.glob("safeview-bundle-*"))
    assert fixture.private_marker not in "\n".join(
        path.read_text(errors="ignore") for path in output.rglob("*") if path.is_file()
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publisher.prepare_bundle(fixture.release, fixture.evaluation, output)
    unreviewed = output / "model/unreviewed_raw_comments.csv"
    unreviewed.write_text(fixture.private_marker)
    with pytest.raises(ValueError, match="content changed"):
        publisher.publish_bundle(output, "fixture/model", "fixture/space")
    unreviewed.unlink()
    linked = output / "model/linked_raw_comments.csv"
    linked.symlink_to(fixture.release / "raw_comments.csv")
    with pytest.raises(ValueError, match="symlinks"):
        publisher.publish_bundle(output, "fixture/model", "fixture/space")


def test_failed_preparation_leaves_no_publishable_partial_bundle(package_fixture, monkeypatch):
    fixture = package_fixture

    def failing_build(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["invented-build-fixture"])

    monkeypatch.setattr(publisher.subprocess, "run", failing_build)
    with pytest.raises(subprocess.CalledProcessError):
        publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output)
    assert not fixture.output.exists()
    assert not list(fixture.output.parent.glob("safeview-bundle-*"))
    with pytest.raises(ValueError, match="Prepare a verified local bundle"):
        publisher.publish_bundle(fixture.release, "fixture/model", "fixture/space")


def test_missing_deploy_templates_uses_versioned_fallback(package_fixture):
    fixture = package_fixture
    old = publisher.PROJECT / "deploy/hf_space"
    fallback = publisher.PROJECT / "scripts/templates/hf_space"
    fallback.mkdir(parents=True)
    for name in ("app.py", "README.md"):
        (old / name).rename(fallback / name)
    publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output)
    assert (fixture.output / "space/app.py").read_bytes() == (fallback / "app.py").read_bytes()
    assert not (old / "app.py").exists()


def test_reuses_matching_bundle_without_model_load_build_or_runtime_checks(package_fixture, monkeypatch):
    fixture = package_fixture
    publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output)
    before = (fixture.output / "bundle.json").read_bytes()
    fixture.calls.clear()

    def reject_rebuild(*args, **kwargs):
        pytest.fail("A historical bundle must not be rebuilt or deserialized")

    monkeypatch.setattr(publisher, "verify_release", reject_rebuild)
    monkeypatch.setattr(publisher.subprocess, "run", reject_rebuild)
    (publisher.PROJECT / "src/safeview_ml/__init__.py").write_text("# New source snapshot\n")
    monkeypatch.setattr(publisher.platform, "python_version", lambda: "0.0.0")
    output = publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output, reuse_existing=True)
    assert output == fixture.output
    assert publisher.verify_existing_bundle_for_release(fixture.release, fixture.evaluation, output)["model_revision"] == "invented-revision"
    assert (output / "bundle.json").read_bytes() == before
    assert fixture.calls == []


@pytest.mark.parametrize("target", ["metadata", "artifact", "evaluation_metadata", "evaluation_metrics", "evaluation_comparison"])
def test_reuse_rejects_different_requested_release_or_evaluation(package_fixture, target):
    fixture = package_fixture
    publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output)
    if target == "metadata":
        publisher.write_json(fixture.release / "metadata.json", {**fixture.metadata, "model_revision": "another-run"})
    elif target == "artifact":
        (fixture.release / "pipeline.joblib").write_bytes(b"Different invented artifact")
    elif target == "evaluation_metadata":
        publisher.write_json(fixture.evaluation / "metadata.json", {**fixture.final, "model_revision": "another-run"})
    elif target == "evaluation_metrics":
        publisher.write_json(fixture.evaluation / "fixture_family/test_metrics.json", {"changed": True})
    else:
        (fixture.evaluation / "comparison.csv").write_text("changed,data\n")
    with pytest.raises(ValueError, match="does not match the requested"):
        publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output, reuse_existing=True)


def test_reuse_rejects_modified_existing_bundle(package_fixture):
    fixture = package_fixture
    publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output)
    (fixture.output / "space/app.py").write_text("# Unreviewed change\n")
    with pytest.raises(ValueError, match="content changed"):
        publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output, reuse_existing=True)


def test_reuse_existing_prepares_when_missing_but_reuse_only_does_not(package_fixture, monkeypatch):
    fixture = package_fixture
    arguments = ["publish_hf.py", "--release-dir", str(fixture.release),
                 "--evaluation-dir", str(fixture.evaluation), "--output-dir", str(fixture.output)]
    monkeypatch.setattr(publisher.sys, "argv", [*arguments, "--reuse-only"])
    with pytest.raises(FileNotFoundError, match="restore its original bundle"):
        publisher.main()
    assert fixture.calls == []
    assert not fixture.output.exists()
    monkeypatch.setattr(publisher.sys, "argv", [*arguments, "--reuse-existing"])
    publisher.main()
    assert fixture.calls == [fixture.release]
    monkeypatch.setattr(publisher.sys, "argv", [*arguments, "--reuse-only"])
    publisher.main()
    assert fixture.calls == [fixture.release]
