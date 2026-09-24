"""Verify that viewing the notebook cannot start downloads or experiments."""
import base64
from copy import deepcopy
import getpass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
import zipfile

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("notebook_builder_test", ROOT / "scripts/build_notebook.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _configuration_source():
    return next(cell.source for cell in builder.build().cells
                if "safeview-configuration" in cell.metadata.get("tags", []))


def _read_only_setup(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("SAFEVIEW_NOTEBOOK_READ_ONLY", "1")
    namespace = {}
    exec(compile(_configuration_source(), "notebook-setup-test", "exec"), namespace)
    return namespace


def test_saved_notebook_workflow_matches_builder_and_accepts_user_configuration():
    notebook = nbformat.read(ROOT / "notebooks/cs114_safeview.ipynb", as_version=4)
    nbformat.validate(notebook)
    expected = builder.build()
    def workflow(cells):
        return [(c.cell_type, c.source) for c in cells
                if not set(c.metadata.get("tags", [])) & {"safeview-configuration", "safeview-study-configuration"}]

    assert workflow(notebook.cells) == workflow(expected.cells)
    assert sum("safeview-configuration" in cell.metadata.get("tags", [])
               for cell in notebook.cells) == 1
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook", "exec")


def test_refresh_preserves_entire_user_configuration_and_unchanged_outputs():
    existing = builder.build()
    configuration = next(cell for cell in existing.cells
                         if "safeview-configuration" in cell.metadata.get("tags", []))
    configuration.source = configuration.source.replace("RUN_SETUP = False", "RUN_SETUP = True")
    configuration.metadata["user-note"] = "Keep every setting and saved diagnostic"
    configuration.execution_count = 4
    configuration.outputs = [nbformat.v4.new_output("stream", name="stdout", text="User output\n")]
    other = next(cell for cell in existing.cells
                 if cell.cell_type == "code" and cell is not configuration)
    other.execution_count = 5
    other.outputs = [nbformat.v4.new_output("stream", name="stdout", text="Saved result\n")]
    before = deepcopy(existing)

    refreshed = builder.refresh_notebook(existing)

    assert existing == before  # Refresh must not mutate its input either.
    assert next(cell for cell in refreshed.cells
                if "safeview-configuration" in cell.metadata.get("tags", [])) == configuration
    assert next(cell for cell in refreshed.cells if cell.id == other.id) == other
    nbformat.validate(refreshed)


def test_refresh_preserves_supplementary_configuration():
    existing = builder.build()
    settings = next(c for c in existing.cells
                    if "safeview-study-configuration" in c.metadata.get("tags", []))
    settings.source = settings.source.replace('REFERENCE_RUN_ID = "vihsd-002"', 'REFERENCE_RUN_ID = "my-reference"')
    settings.source = settings.source.replace('IMBALANCE_FAMILIES = ["svm", "logistic_regression", "phobert", "bamibert"]', 'IMBALANCE_FAMILIES = ["bamibert"]')
    settings.execution_count = 9
    refreshed = builder.refresh_notebook(existing)
    assert next(c for c in refreshed.cells
                if "safeview-study-configuration" in c.metadata.get("tags", [])) == settings


def test_supplementary_mode_does_not_execute_old_action_flags(tmp_path, monkeypatch):
    import safeview_ml.training
    notebook = builder.build()
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical run must never be trained, tested or bundled in study mode")
    monkeypatch.setattr(safeview_ml.training, "train_experiment", forbidden)
    monkeypatch.setattr(safeview_ml.training, "evaluate_locked", forbidden)
    namespace = {
        "IMBALANCE_STUDY": True, "RUN_TRAINING": True, "RUN_FINAL_TEST": True,
        "RUN_PREPARE_BUNDLE": True, "RUN_ROOT": tmp_path, "RUN_ID": "old-run",
        "RUN": None, "bundle": None, "train_experiment": forbidden,
        "evaluate_locked": forbidden, "subprocess": type("Forbidden", (), {"run": forbidden}),
    }
    prefixes = ("if RUN_TRAINING and not IMBALANCE_STUDY:", "FINAL = ", "BUNDLE = ")
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.startswith(prefixes):
            exec(compile(cell.source, "old-run-protection", "exec"), namespace)
    assert list(tmp_path.iterdir()) == []


def test_supplementary_read_only_disables_derived_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("SAFEVIEW_NOTEBOOK_READ_ONLY", "1")
    settings = next(c.source for c in builder.build().cells
                    if "safeview-study-configuration" in c.metadata.get("tags", []))
    namespace = {"os": os, "RUN_ID": "vihsd-003", "RUN_ROOT": tmp_path,
                 "RUN_TRAINING": True, "RUN_FINAL_TEST": True, "RUN_PREPARE_BUNDLE": True}
    exec(compile(settings, "study-read-only", "exec"), namespace)
    assert all(namespace[flag] is False for flag in ("STUDY_TRAIN", "STUDY_TEST", "STUDY_BUNDLE"))


def test_migrate_old_study_controls_preserves_primary_cell_and_family_choices():
    notebook = builder.build()
    primary = next(c for c in notebook.cells if "safeview-configuration" in c.metadata.get("tags", []))
    primary.source = primary.source.replace('RUN_ID = os.getenv("SAFEVIEW_RUN_ID") or None', 'RUN_ID = "vihsd-003"')
    primary_before = deepcopy(primary)
    settings = next(c for c in notebook.cells if "safeview-study-configuration" in c.metadata.get("tags", []))
    settings.metadata.pop("safeview-study-schema")
    settings.source = '''IMBALANCE_STUDY = True
IMBALANCE_REFERENCE_RUN_ID = RUN_ID or "vihsd-002"
IMBALANCE_STUDY_ID = IMBALANCE_REFERENCE_RUN_ID + "-imbalance-01"
IMBALANCE_FAMILIES = ["bamibert"]
RUN_IMBALANCE_TRAINING = False
'''
    refreshed = builder.refresh_notebook(notebook)
    assert next(c for c in refreshed.cells if "safeview-configuration" in c.metadata.get("tags", [])) == primary_before
    migrated = next(c for c in refreshed.cells if "safeview-study-configuration" in c.metadata.get("tags", []))
    assert migrated.metadata["safeview-study-schema"] == 3
    assert 'REFERENCE_RUN_ID = "vihsd-002"' in migrated.source
    assert "IMBALANCE_FAMILIES = ['bamibert']" in migrated.source
    assert "RUN_IMBALANCE_" not in migrated.source


@pytest.mark.parametrize("training", [True, False])
def test_simple_run_uses_new_id_and_original_flags(tmp_path, monkeypatch, training):
    from types import SimpleNamespace
    import safeview_ml.imbalance as study_api
    monkeypatch.delenv("SAFEVIEW_NOTEBOOK_READ_ONLY", raising=False)
    notebook = builder.build()
    settings = next(c.source for c in notebook.cells if "safeview-study-configuration" in c.metadata.get("tags", []))
    actions = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("from safeview_ml.imbalance import"))
    calls = []
    study = tmp_path / "results/studies/vihsd-003"
    study.mkdir(parents=True)
    (study / "protocol.json").write_text("{}")

    def prepare(project, reference, run_id):
        calls.append(("prepare", reference, run_id))
        assert project == tmp_path
        return study

    def train(bundle, path, *, families, resume):
        calls.append(("train", path.name, resume))
        assert families == ["svm", "logistic_regression", "phobert", "bamibert"]

    def evaluate(bundle, path):
        calls.append(("test", path.name))

    import pandas as pd
    monkeypatch.setattr(study_api, "prepare_imbalance_study", prepare)
    monkeypatch.setattr(study_api, "train_imbalance_study", train)
    monkeypatch.setattr(study_api, "evaluate_imbalance_study", evaluate)
    monkeypatch.setattr(study_api, "export_imbalance_comparison", lambda *a, **k: pd.DataFrame())
    namespace = {"os": os, "RUN_ID": "vihsd-003", "RUN_ROOT": tmp_path,
                 "RUN_TRAINING": training, "RESUME_TRAINING": training,
                 "RUN_FINAL_TEST": True, "RUN_PREPARE_BUNDLE": True,
                 "bundle": SimpleNamespace(manifest={"synthetic": False}),
                 "display": lambda *a: None, "read_json": lambda p: {}, "pd": pd}
    exec(compile(settings, "simple-run-settings", "exec"), namespace)
    exec(compile(actions, "simple-run-actions", "exec"), namespace)
    if training:
        assert calls == [("prepare", "vihsd-002", "vihsd-003"), ("train", "vihsd-003", True), ("test", "vihsd-003")]
        assert namespace["STUDY_TEST"] is namespace["STUDY_BUNDLE"] is True
    else:
        assert calls == [("test", "vihsd-003")]
        assert namespace["STUDY_TRAIN"] is False


@pytest.mark.parametrize("historical_wins", [False, True])
def test_run_all_orders_training_test_and_local_bundle(tmp_path, monkeypatch, historical_wins):
    """All original flags may stay true; no inference or publishing in this test."""
    from types import SimpleNamespace
    import pandas as pd
    import safeview_ml.imbalance as study_api
    from safeview_ml.provenance import read_json, write_json
    monkeypatch.delenv("SAFEVIEW_NOTEBOOK_READ_ONLY", raising=False)
    notebook = builder.build()
    settings = next(c.source for c in notebook.cells if "safeview-study-configuration" in c.metadata.get("tags", []))
    actions = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("from safeview_ml.imbalance import"))
    packaging = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("if STUDY_BUNDLE:"))
    study = tmp_path / "results/studies/vihsd-003"
    selected = "vihsd-002" if historical_wins else "vihsd-003-bamibert-balanced"
    calls = []

    def prepare(*args):
        calls.append("prepare")
        write_json(study / "protocol.json", {})
        return study

    def train(*args, **kwargs):
        calls.append("train_all")

    def evaluate(*args):
        assert calls[-1] == "train_all"
        calls.append("lock_and_test")
        write_json(study / "study_selection.json", {"validation_best_by_family": {"bamibert": {"run_id": selected}}})
        write_json(tmp_path / "results/final" / selected / "metadata.json", {"status": "evaluated"})

    def local_bundle(command, **kwargs):
        assert calls[-1] == "lock_and_test"
        calls.append("bundle")
        assert command[-1] == ("--reuse-only" if historical_wins else "--reuse-existing")
        assert "--publish" not in command
        assert command[command.index("--release-dir") + 1] == str(tmp_path / "artifacts" / selected / "bamibert")

    monkeypatch.setattr(study_api, "prepare_imbalance_study", prepare)
    monkeypatch.setattr(study_api, "train_imbalance_study", train)
    monkeypatch.setattr(study_api, "evaluate_imbalance_study", evaluate)
    monkeypatch.setattr(study_api, "export_imbalance_comparison", lambda *a, **k: pd.DataFrame())
    namespace = {"os": os, "ROOT": ROOT, "sys": sys, "RUN_ROOT": tmp_path, "RUN_ID": "vihsd-003",
                 "RUN_TRAINING": True, "RESUME_TRAINING": False, "RUN_FINAL_TEST": True, "RUN_PREPARE_BUNDLE": True,
                 "bundle": SimpleNamespace(manifest={"synthetic": False}), "read_json": read_json, "pd": pd,
                 "display": lambda *a: None, "Markdown": lambda x: x,
                 "subprocess": SimpleNamespace(run=local_bundle)}
    for source in (settings, actions, packaging):
        exec(compile(source, "run-all-order", "exec"), namespace)
    assert calls == ["prepare", "train_all", "lock_and_test", "bundle"]


def test_migrates_priority_controls_to_sequential_run_all():
    notebook = builder.build()
    settings = next(c for c in notebook.cells if "safeview-study-configuration" in c.metadata.get("tags", []))
    settings.metadata["safeview-study-schema"] = 2
    settings.source = settings.source.replace("and not _study_read_only", "and not RUN_TRAINING and not _study_read_only")
    settings.source = settings.source.replace('REFERENCE_RUN_ID = "vihsd-002"', 'REFERENCE_RUN_ID = "custom-reference"')
    refreshed = builder.refresh_notebook(notebook)
    migrated = next(c for c in refreshed.cells if "safeview-study-configuration" in c.metadata.get("tags", []))
    assert migrated.metadata["safeview-study-schema"] == 3
    assert "and not RUN_TRAINING" not in migrated.source
    assert "REFERENCE_RUN_ID = 'custom-reference'" in migrated.source


def test_simple_run_reads_reference_revision_without_changing_new_id(tmp_path, monkeypatch):
    import safeview_ml.provenance as provenance
    from safeview_ml.provenance import write_json
    import IPython.display
    notebook = builder.build()
    imports = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("import json\n"))
    reference = tmp_path / "results/runs/vihsd-002"
    write_json(reference / "metadata.json", {"dataset_manifest": {
        "source": "https://github.com/sonlam1102/vihsd", "revision": "a" * 40}})
    monkeypatch.setattr(provenance, "environment_metadata", lambda: {"python": "fixture", "platform": "offline", "versions": {}})
    monkeypatch.setattr(IPython.display, "display", lambda *a: None)
    namespace = {"ROOT": ROOT, "RUN_ROOT": tmp_path, "CONFIG_NAME": "experiments.yaml",
                 "IMBALANCE_STUDY": True, "REFERENCE_RUN_ID": "vihsd-002", "RUN_ID": "vihsd-003",
                 "RUN_TRAINING": False, "DATA_REVISION": "main"}
    exec(compile(imports, "reference-revision", "exec"), namespace)
    assert namespace["RUN_ID"] == "vihsd-003"
    assert namespace["RUN"] == reference
    assert namespace["VIEW_RUN_ID"] == "vihsd-002"
    assert namespace["DATA_REVISION"] == "a" * 40


def test_refresh_drops_outputs_only_when_generated_source_changes():
    existing = builder.build()
    other = next(cell for cell in existing.cells
                 if cell.cell_type == "code" and not set(cell.metadata.get("tags", [])) &
                 {"safeview-configuration", "safeview-study-configuration"})
    original_source = other.source
    other.source += "\n# An older generated version"
    other.execution_count = 3
    other.outputs = [nbformat.v4.new_output("stream", name="stdout", text="Stale result\n")]

    refreshed = builder.refresh_notebook(existing)

    replacement = next(cell for cell in refreshed.cells if cell.source == original_source)
    assert replacement.execution_count is None
    assert replacement.outputs == []


@pytest.mark.parametrize("count", [0, 2])
def test_refresh_refuses_ambiguous_configuration(count):
    existing = builder.build()
    configuration = next(cell for cell in existing.cells
                         if "safeview-configuration" in cell.metadata.get("tags", []))
    if count == 0:
        configuration.metadata["tags"] = []
    else:
        existing.cells.append(deepcopy(configuration))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        builder.refresh_notebook(existing)


def test_builder_execute_forces_read_only_and_keeps_configuration(tmp_path, monkeypatch):
    import jupyter_client
    import nbclient
    from types import SimpleNamespace

    existing = builder.build()
    configuration = next(cell for cell in existing.cells
                         if "safeview-configuration" in cell.metadata.get("tags", []))
    for flag in builder.ACTION_FLAGS:
        configuration.source = re.sub(rf"^{flag} = False", f"{flag} = True",
                                      configuration.source, flags=re.MULTILINE)
    configuration.outputs = [nbformat.v4.new_output("stream", name="stdout", text="Keep me\n")]
    configuration.execution_count = 12
    path = tmp_path / "notebooks/cs114_safeview.ipynb"
    path.parent.mkdir()
    nbformat.write(existing, path)

    class FakeClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook

        def execute(self, *, env):
            assert env["SAFEVIEW_NOTEBOOK_READ_ONLY"] == "1"
            for cell in self.notebook.cells:
                if cell.cell_type == "code":
                    cell.execution_count = 99
                    cell.outputs = []
            return self.notebook

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    runtime_source = next(cell.source for cell in existing.cells
                          if "safeview-runtime-sync" in cell.metadata.get("tags", []))
    monkeypatch.setattr(builder, "runtime_sync_source", lambda: runtime_source)
    monkeypatch.setattr(sys, "argv", ["build_notebook.py", "--execute"])
    monkeypatch.setattr(jupyter_client, "KernelManager",
                        lambda **kwargs: SimpleNamespace(kernel_spec=SimpleNamespace(argv=["python"])))
    monkeypatch.setattr(nbclient, "NotebookClient", FakeClient)

    builder.main()

    saved = nbformat.read(path, as_version=4)
    assert next(cell for cell in saved.cells
                if "safeview-configuration" in cell.metadata.get("tags", [])) == configuration


def test_readonly_execution_disables_even_previously_enabled_flags(tmp_path, monkeypatch):
    import IPython.display
    import safeview_ml.provenance
    import safeview_ml.remote_data
    import safeview_ml.training
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only notebook attempted network/setup/training/test")

    notebook = builder.build()
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("SAFEVIEW_NOTEBOOK_READ_ONLY", "1")
    monkeypatch.setenv("SAFEVIEW_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("SAFEVIEW_RUN_ID", "")
    monkeypatch.setattr(getpass, "getpass", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(safeview_ml.remote_data, "load_github_dataset", forbidden)
    monkeypatch.setattr(safeview_ml.training, "train_experiment", forbidden)
    monkeypatch.setattr(safeview_ml.training, "evaluate_locked", forbidden)
    monkeypatch.setattr(safeview_ml.provenance, "environment_metadata", lambda: {
        "python": "software-fixture", "platform": "offline", "versions": {},
    })
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if "safeview-configuration" in cell.metadata.get("tags", []):
            for flag in builder.ACTION_FLAGS:
                source = re.sub(rf"^{flag} = False", f"{flag} = True", source, flags=re.MULTILINE)
        exec(compile(source, "read-only-notebook-test", "exec"), namespace)
    assert all(namespace[flag] is False for flag in builder.ACTION_FLAGS)
    assert list(tmp_path.iterdir()) == []


def test_private_clone_authentication_is_scoped_to_subprocess(monkeypatch, tmp_path, capsys):
    namespace = _read_only_setup(monkeypatch)
    token = "github_test_token_do_not_log"
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    prompts = []
    calls = []

    def prompt(message):
        prompts.append(message)
        return token

    def run(command, **kwargs):
        kwargs["env"] = dict(kwargs["env"])
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="Cloning into checkout...\n")

    monkeypatch.setattr(getpass, "getpass", prompt)
    monkeypatch.setattr(subprocess, "run", run)
    original_environment = dict(os.environ)
    repo_url = namespace["REPO_URL"]
    namespace["run_git"]("clone", "--no-checkout", repo_url,
                         str(tmp_path / "checkout"), authenticate=True)

    assert len(prompts) == len(calls) == 1
    command, options = calls[0]
    assert command == ["git", "clone", "--no-checkout", repo_url, str(tmp_path / "checkout")]
    assert token not in repr(command) and encoded not in repr(command)
    assert options["stdout"] == subprocess.PIPE
    assert options["stderr"] == subprocess.STDOUT
    assert options["text"] is True
    git_environment = options["env"]
    config = {git_environment[f"GIT_CONFIG_KEY_{index}"]:
              git_environment[f"GIT_CONFIG_VALUE_{index}"]
              for index in range(int(git_environment["GIT_CONFIG_COUNT"]))}
    header, authorization, credentials = config[f"http.{repo_url}.extraheader"].split()
    assert header.lower() == "authorization:"
    assert authorization.lower() == "basic"
    assert credentials == encoded
    assert dict(os.environ) == original_environment
    output = capsys.readouterr().out
    assert token not in output and encoded not in output


def test_failed_clone_shows_git_error_without_credentials(monkeypatch, tmp_path, capsys):
    namespace = _read_only_setup(monkeypatch)
    token = "github_test_secret_for_redaction"
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    monkeypatch.setattr(getpass, "getpass", lambda message: token)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 128, stdout=(
            "fatal: repository access denied\n"
            f"token={token}\nAuthorization: Basic {encoded}\n"))

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError) as error:
        namespace["run_git"]("clone", "--no-checkout", namespace["REPO_URL"],
                             str(tmp_path / "checkout"), authenticate=True)

    message = str(error.value)
    captured = capsys.readouterr()
    assert "repository access denied" in message
    assert "128" in message
    for visible_output in (message, captured.out, captured.err):
        assert token not in visible_output
        assert encoded not in visible_output


def test_existing_checkout_is_verified_without_authentication(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("", encoding="utf-8")
    source = _configuration_source()
    source = re.sub(r"^RUN_SETUP = False", "RUN_SETUP = True", source, flags=re.MULTILINE)
    source = re.sub(r"^COLAB_REPO = .+$", f"COLAB_REPO = Path({str(checkout)!r})",
                    source, flags=re.MULTILINE)
    calls = []
    repo_url = "https://github.com/andrewthongle/cs114-ml-project.git"
    commit = "a" * 40

    def forbidden_prompt(*args, **kwargs):
        raise AssertionError("An existing checkout must not ask for GitHub credentials")

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "git":
            assert command[:3] == ["git", "-C", str(checkout)]
            if command[3:] == ["remote", "get-url", "origin"]:
                output = repo_url + "\n"
            else:
                assert command[3:5] == ["rev-parse", "HEAD"] or command[3:] == [
                    "rev-parse", "--verify", "--end-of-options", "main^{commit}"]
                output = commit + "\n"
            return subprocess.CompletedProcess(command, 0, stdout=output)
        assert command[:5] == [sys.executable, "-m", "pip", "install", "-e"]
        assert command[5] == str(checkout) + "[dev,hub,serve,transformers]"
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("SAFEVIEW_NOTEBOOK_READ_ONLY", "0")
    monkeypatch.setitem(sys.modules, "google.colab", ModuleType("google.colab"))
    monkeypatch.setattr(getpass, "getpass", forbidden_prompt)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {}
    exec(compile(source, "existing-checkout-test", "exec"), namespace)

    assert namespace["ROOT"] == checkout
    assert len([command for command in calls if command[0] == "git"]) == 3
    assert all("clone" not in command for command in calls)


def test_embedded_sync_repairs_old_colab_api_and_reloads_cached_modules(tmp_path):
    """Reproduce the user's old Python API + old YAML inside a separate kernel."""
    notebook = builder.build()
    sync = next(c.source for c in notebook.cells if "safeview-runtime-sync" in c.metadata.get("tags", []))
    imports = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("import json\n"))
    import ast
    tree = ast.parse(sync)
    payload = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_runtime_payload")
    digest = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_runtime_digest")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["manifest.json"])
    for record in manifest["files"]:
        path = tmp_path / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = contents[record["path"]]
        if record["path"] == "src/safeview_ml/data.py":
            previous = b"def validate_training_data(bundle):\n    return None\n"
        if record["path"] == "configs/experiments.yaml":
            previous = previous.replace(b"empty_text_policy: keep\n", b"")
        path.write_bytes(previous)
        record["previous_sha256"] = hashlib.sha256(previous).hexdigest()
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    contents["manifest.json"] = json.dumps(manifest).encode()
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    sync = sync.replace(repr(payload), repr(base64.b64encode(packed.getvalue()).decode()))
    sync = sync.replace(repr(digest), repr(hashlib.sha256(packed.getvalue()).hexdigest()))
    runner = f'''
import inspect, os, sys
from pathlib import Path
ROOT = Path({str(tmp_path)!r})
sys.path.insert(0, str(ROOT / "src"))
import safeview_ml.data as cached_old_data
assert "empty_text_policy" not in inspect.signature(cached_old_data.validate_training_data).parameters
IN_COLAB = RUN_SETUP = True
os.environ.pop("SAFEVIEW_NOTEBOOK_READ_ONLY", None)
exec({sync!r})
RUN_ROOT = ROOT / "outputs"
RUN_ID = "synthetic-sync-check"
DATA_REVISION = "main"
CONFIG_NAME = "experiments.yaml"
RUN_TRAINING = False
exec({imports!r})
assert CONFIG["empty_text_policy"] == "keep"
assert "empty_text_policy" in inspect.signature(validate_training_data).parameters
assert validate_training_data is not cached_old_data.validate_training_data
from safeview_ml.data import bundle_from_frames
frames = {{split: pd.DataFrame({{"text": ["", " ", "fixture"], "label": [0, 1, 2]}})
          for split in ("train", "validation", "test")}}
bundle = bundle_from_frames(frames, source_metadata={{"synthetic": True}})
assert validate_training_data(bundle, empty_text_policy=CONFIG["empty_text_policy"])["empty_text_counts"]["train"] == 2
assert not RUN_ROOT.exists()
'''
    result = subprocess.run([sys.executable, "-c", runner], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
