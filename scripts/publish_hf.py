#!/usr/bin/env python3
"""Prepare an auditable bundle, then optionally publish to explicit HF repos.

Preparation never contacts Hugging Face. Synthetic checks cannot be published.
Run with the same environment and source snapshot used by training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

from safeview_ml.inference import ReleasePredictor, file_sha256, release_file_names

PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def space_template_file(name: str) -> Path:
    """Prefer local serving overrides, with checked-in templates as fallback."""
    if name not in {"app.py", "README.md"}:
        raise ValueError(f"Unsupported Space template: {name}")
    for folder in (PROJECT / "deploy/hf_space", PROJECT / "scripts/templates/hf_space"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing Space template {name}; restore scripts/templates/hf_space before packaging")


def evaluation_export_files(evaluation: Path):
    """The closed set of evaluation reports that may enter a bundle."""
    return [evaluation / "metadata.json", evaluation / "comparison.csv",
            *sorted(evaluation.glob("*/test_metrics.json"))]


def verify_release(release: Path, evaluation: Path):
    """Check provenance before joblib deserialization or building a wheel."""
    metadata = read_json(release / "metadata.json")
    final = read_json(evaluation / "metadata.json")
    if metadata.get("synthetic") is not False or final.get("synthetic") is not False:
        raise ValueError("Only a real dataset release may be packaged; synthetic/unknown provenance rejected")
    if final.get("status") != "evaluated":
        raise ValueError("Complete the frozen final test evaluation before packaging")
    if final.get("model_revision") != metadata.get("model_revision"):
        raise ValueError("Final evaluation does not describe this model revision")
    if final.get("dataset_fingerprint") != metadata.get("dataset_fingerprint"):
        raise ValueError("Final evaluation dataset fingerprint differs from the release")
    hashes = metadata.get("sha256", {})
    for name in set(release_file_names(metadata)) - {"metadata.json"}:
        digest = file_sha256(release / name)
        if hashes.get(name) != digest or final.get("artifact_sha256", {}).get(name) != digest:
            raise ValueError(f"Release differs from evaluated artifact: {name}")
    expected_sources = metadata.get("source_sha256", {})
    actual_sources = {
        str(path.relative_to(PROJECT)): file_sha256(path)
        for path in sorted((PROJECT / "src/safeview_ml").glob("*.py"))
    }
    actual_sources["pyproject.toml"] = file_sha256(PROJECT / "pyproject.toml")
    if not expected_sources or expected_sources != actual_sources:
        raise ValueError("Training source differs from current source; restore its snapshot before packaging")
    if metadata.get("python") != platform.python_version():
        raise ValueError("Build with the Python version recorded by training")
    for name, expected in metadata.get("versions", {}).items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Installed {name} differs from the training environment")
    return ReleasePredictor(release), final


def pinned_dependencies(extra=None):
    """Freeze the dependency closure for the CPU Space.

    Evaluate markers for Linux/Python matching the Space. This refuses to invent
    a version if a Linux dependency is not installed in the build environment.
    Transformer releases use the same PyTorch version's CPU wheel, so a Colab
    CUDA installation does not add CUDA libraries to the CPU Space.
    """
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    project = tomllib.loads((PROJECT / "pyproject.toml").read_text())["project"]
    requirements = project["dependencies"] + project["optional-dependencies"]["serve"]
    if extra:
        requirements += project["optional-dependencies"][extra]
    environment = default_environment()
    environment.update(sys_platform="linux", platform_system="Linux", extra="")
    pending = [Requirement(value) for value in requirements]
    pinned = {}
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        if extra == "transformers" and (name.startswith(("nvidia-", "cuda-")) or name == "triton"):
            continue
        if name in pinned:
            continue
        distribution = importlib.metadata.distribution(name)
        version = distribution.version
        if extra == "transformers" and name == "torch":
            version = version.split("+", 1)[0] + "+cpu"
        pinned[name] = version
        pending.extend(Requirement(value) for value in distribution.requires or [])
    indexes = ["--extra-index-url https://download.pytorch.org/whl/cpu"] if extra == "transformers" else []
    return indexes + [f"{name}=={version}" for name, version in sorted(pinned.items())]


def bundle_file_digests(bundle: Path):
    """Inventory only the two folders sent to HF, refusing linked-in content."""
    digests = {}
    for name in ("model", "space"):
        folder = bundle / name
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Prepared bundle requires an ordinary {name} directory")
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError("Prepared bundle must not contain symlinks")
            if path.is_file():
                digests[path.relative_to(bundle).as_posix()] = file_sha256(path)
    return digests


def verify_prepared_bundle(bundle: Path):
    """Detect changed or added upload content before any publication request."""
    bundle = Path(bundle)
    if not (bundle / "bundle.json").is_file():
        raise ValueError("Prepare a verified local bundle before publishing")
    manifest = read_json(bundle / "bundle.json")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict) or not expected or expected != bundle_file_digests(bundle):
        raise ValueError("Prepared bundle content changed; prepare a new bundle before publishing")
    metadata = read_json(bundle / "model/metadata.json")
    final = read_json(bundle / "model/evaluation/metadata.json")
    if metadata.get("synthetic") is not False or final.get("synthetic") is not False:
        raise ValueError("Synthetic/unknown provenance cannot be published")
    if final != manifest.get("evaluation") or final.get("status") != "evaluated":
        raise ValueError("Prepared bundle requires its completed frozen evaluation")
    if not manifest.get("model_revision") or not (
        manifest["model_revision"] == metadata.get("model_revision") == final.get("model_revision")
    ):
        raise ValueError("Prepared bundle model revisions differ")
    if not metadata.get("dataset_fingerprint") or final.get("dataset_fingerprint") != metadata["dataset_fingerprint"]:
        raise ValueError("Prepared bundle evaluation dataset fingerprint differs from the release")
    for name in set(release_file_names(metadata)) - {"metadata.json"}:
        digest = expected.get(f"model/{name}")
        if digest != metadata.get("sha256", {}).get(name) or digest != final.get("artifact_sha256", {}).get(name):
            raise ValueError("Prepared bundle differs from its evaluated artifact")
    return manifest


def verify_existing_bundle_for_release(release_dir: Path, evaluation_dir: Path, bundle_dir: Path):
    """Reuse only an intact bundle of the exact requested release/evaluation.

    This checks original bytes without loading the model, rebuilding its wheel,
    or imposing today's source/runtime on a previously evaluated release.
    """
    release_dir, evaluation_dir, bundle_dir = map(Path, (release_dir, evaluation_dir, bundle_dir))
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"No existing verified bundle at {bundle_dir}; restore its original bundle "
            "to reuse a historical release without rebuilding it"
        )
    manifest = verify_prepared_bundle(bundle_dir)
    metadata = read_json(release_dir / "metadata.json")
    requested = {
        f"model/{name}": file_sha256(release_dir / name)
        for name in release_file_names(metadata)
    }
    evaluation_files = {
        f"model/evaluation/{path.relative_to(evaluation_dir).as_posix()}": file_sha256(path)
        for path in evaluation_export_files(evaluation_dir)
    }
    expected = manifest["files_sha256"]
    bundled_evaluation = {name: digest for name, digest in expected.items()
                          if name.startswith("model/evaluation/")}
    if evaluation_files != bundled_evaluation:
        raise ValueError("Existing bundle does not match the requested frozen evaluation")
    if any(expected.get(name) != digest for name, digest in requested.items()):
        raise ValueError("Existing bundle does not match the requested release")
    return manifest


def prepare_bundle(release_dir: Path, evaluation_dir: Path, output_dir: Path, *, reuse_existing=False):
    release_dir, evaluation_dir, output_dir = map(Path, (release_dir, evaluation_dir, output_dir))
    if output_dir.exists():
        if reuse_existing:
            verify_existing_bundle_for_release(release_dir, evaluation_dir, output_dir)
            return output_dir
        raise FileExistsError(f"Refusing to overwrite bundle: {output_dir}")
    app_template, readme_template = space_template_file("app.py"), space_template_file("README.md")
    predictor, final = verify_release(release_dir, evaluation_dir)
    is_transformer = predictor.metadata.get("backend") == "transformers"
    dependencies = pinned_dependencies("transformers") if is_transformer else pinned_dependencies()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="safeview-bundle-", dir=output_dir.parent) as staging:
        stage = Path(staging)
        model, space = stage / "model", stage / "space"
        model.mkdir()
        space.mkdir()
        for name in release_file_names(predictor.metadata):
            shutil.copy2(release_dir / name, model / name)
        (model / "evaluation").mkdir()
        for path in evaluation_export_files(evaluation_dir):
            destination = model / "evaluation" / path.relative_to(evaluation_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        library = "transformers" if is_transformer else "sklearn"
        base_model = predictor.metadata.get("base_model")
        base_header = f"base_model: {base_model}\n" if base_model else ""
        if final.get("selection_criterion") == "predeclared_deployment_family":
            selection_text = (
                f"Deployment family `{final['deployment_family']}` was declared before training. "
                f"The validation Macro-F1 winner is `{final['validation_best_family']}`. "
                "Deployment selection does not by itself establish superior accuracy. "
            )
        else:
            selection_text = "The deployment model was selected using validation Macro-F1. "
        card = (
            f"---\nlanguage: vi\nlibrary_name: {library}\npipeline_tag: text-classification\n{base_header}tags:\n- text-classification\n- research\n---\n\n"
            "# SafeView CS114 research classifier\n\n"
            f"Model revision: `{predictor.model_revision}`. Policy version: `{predictor.policy_version}`.\n\n"
            f"{selection_text}The linked evaluation contains the test results for "
            "all frozen candidates.\n\n"
            "Input is one Vietnamese comment. Labels follow ViHSD: CLEAN (0), "
            "OFFENSIVE (1), HATE (2). Three-class prediction uses argmax. The "
            "independent hiding decision uses P(OFFENSIVE)+P(HATE) >= the locked "
            "threshold in decision_policy.json. Confidence is not severity.\n\n"
            "Source: [ViHSD authors](https://github.com/sonlam1102/vihsd), "
            "[UIT dataset](https://huggingface.co/datasets/uitnlp/vihsd). Dataset "
            "revision, fingerprint, feature configuration, runtime versions and "
            "source digests are recorded in metadata.json. Raw examples are not "
            "included. The dataset's research restrictions and conflicting license "
            "metadata require review before public or commercial release.\n\n"
            "The classifier lacks conversation context and may fail on irony, "
            "quoted abuse, dialect, teencode and domain shift. It is a research "
            "component, not an exhaustive content-safety system.\n\n"
            "Install the bundled wheel and "
            "requirements.txt using the Python version in metadata.json, then call "
            "`safeview_ml.inference.ReleasePredictor(release_dir)`. See "
            "[test comparison](evaluation/comparison.csv) and the per-model test "
            "metrics under evaluation/.\n"
        )
        if is_transformer:
            card += (
                "\n## Transformer input and upstream terms\n\n"
                "Use `ReleasePredictor` to reproduce the saved preprocessing and truncation. "
                "A generic Transformers pipeline alone does not perform this release's "
                "PhoBERT word segmentation or apply its hiding threshold. "
                "PhoBERT uses PyVi segmentation; BamiBERT consumes raw text. "
                "The exact settings and upstream revision are in transformer_config.json.\n\n"
                "Upstream: [PhoBERT](https://huggingface.co/vinai/phobert-base) and "
                "[BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT). "
                "Retain the applicable upstream license and citation notices. BamiBERT "
                "is released under BSD-3-Clause-Clear and Qualcomm Responsible AI terms. "
                "Fine-tuning does not remove upstream or dataset conditions.\n"
            )
        (model / "README.md").write_text(card, encoding="utf-8")
        wheel_dir = stage / "wheels"
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheel_dir.resolve())],
            cwd=PROJECT, check=True,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected exactly one project wheel")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            packaged_sources = {
                f"src/{name}": hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if name.startswith("safeview_ml/") and name.endswith(".py")
            }
        expected_sources = {name: digest for name, digest in predictor.metadata["source_sha256"].items()
                            if name.startswith("src/")}
        if packaged_sources != expected_sources:
            raise ValueError("Built wheel does not match the evaluated source snapshot")
        for folder in (model, space):
            shutil.copy2(wheel, folder / wheel.name)
            (folder / "requirements.txt").write_text(
                "\n".join([f"./{wheel.name}", *dependencies]) + "\n", encoding="utf-8"
            )
        shutil.rmtree(wheel_dir)
        shutil.copy2(app_template, space / "app.py")
        readme = readme_template.read_text(encoding="utf-8")
        readme = readme.replace(
            "This directory is a source template, **not a published or evaluated model**.",
            f"This bundle serves evaluated model `{predictor.model_revision}` with policy `{predictor.policy_version}`.",
        )
        readme = readme.replace("sdk: gradio\n", f"sdk: gradio\nsdk_version: {importlib.metadata.version('gradio')}\n")
        readme = re.sub(r'^python_version:.*$', f'python_version: "{platform.python_version()}"', readme, flags=re.MULTILINE)
        (space / "README.md").write_text(readme, encoding="utf-8")
        write_json(stage / "bundle.json", {
            "model_revision": predictor.model_revision,
            "policy_version": predictor.policy_version,
            "evaluation": final,
            "wheel_sha256": file_sha256(model / wheel.name),
            "source_sha256": predictor.metadata["source_sha256"],
            "publish_status": "prepared_locally",
            "serving_device": "cpu",
            "pytorch_cpu_build": is_transformer,
            "files_sha256": bundle_file_digests(stage),
        })
        # Rename only after every check and copy succeeds; an incomplete bundle
        # is never mistaken for a reviewable publication directory.
        stage.rename(output_dir)
    return output_dir


def publish_bundle(bundle: Path, model_repo: str, space_repo: str, *, public=False):
    bundle = Path(bundle)
    for repo in (model_repo, space_repo):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("Explicit Hugging Face owner/repository IDs are required")
    manifest = verify_prepared_bundle(bundle)
    from huggingface_hub import HfApi

    api = HfApi()  # HF_TOKEN/login only; never print or copy authentication.
    api.create_repo(repo_id=model_repo, repo_type="model", private=not public, exist_ok=True)
    verify_prepared_bundle(bundle)
    commit = api.upload_folder(
        repo_id=model_repo, repo_type="model", folder_path=bundle / "model",
        commit_message="Publish evaluated CS114 pipeline and frozen policy",
    )
    write_json(bundle / "space/deployment.json", {"model_repo": model_repo, "hub_revision": commit.oid})
    # This generated routing file is the sole expected mutation after review.
    manifest["files_sha256"]["space/deployment.json"] = file_sha256(bundle / "space/deployment.json")
    write_json(bundle / "bundle.json", manifest)
    # Preserve partial success if a Space account/compute restriction blocks
    # creation. The uploaded model is not falsely reported as a running API.
    publication = {"model_repo": model_repo, "model_commit": commit.oid, "space_repo": space_repo, "space_status": "not_uploaded"}
    write_json(bundle / "publication.json", publication)
    api.create_repo(repo_id=space_repo, repo_type="space", space_sdk="gradio", private=not public, exist_ok=True)
    verify_prepared_bundle(bundle)
    space_commit = api.upload_folder(
        repo_id=space_repo, repo_type="space", folder_path=bundle / "space",
        commit_message="Serve evaluated CS114 release with pinned package",
    )
    publication.update(space_commit=space_commit.oid, space_status="uploaded_runtime_unverified")
    write_json(bundle / "publication.json", publication)
    return publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-existing", action="store_true",
                       help="Verify and reuse an existing matching bundle; otherwise prepare a new bundle")
    reuse.add_argument("--reuse-only", action="store_true",
                       help="Require an existing matching bundle without rebuilding a historical release")
    parser.add_argument("--publish", action="store_true", help="Upload the freshly prepared bundle")
    parser.add_argument("--model-repo")
    parser.add_argument("--space-repo")
    parser.add_argument("--public", action="store_true", help="Create public rather than private repositories")
    parser.add_argument("--acknowledge-data-rights", action="store_true", help="Confirm model distribution is permitted by your dataset access terms")
    args = parser.parse_args()
    if args.publish and not (args.model_repo and args.space_repo and args.acknowledge_data_rights):
        parser.error("--publish requires explicit --model-repo, --space-repo and --acknowledge-data-rights")
    if args.reuse_only:
        verify_existing_bundle_for_release(args.release_dir, args.evaluation_dir, args.output_dir)
        bundle = args.output_dir
    else:
        bundle = prepare_bundle(args.release_dir, args.evaluation_dir, args.output_dir,
                                reuse_existing=args.reuse_existing)
    print(f"Verified local bundle: {bundle}")
    if args.publish:
        publication = publish_bundle(bundle, args.model_repo, args.space_repo, public=args.public)
        print(json.dumps(publication, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
