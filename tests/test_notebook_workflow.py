"""Verify that viewing the notebook cannot start downloads or experiments."""
import importlib.util
from pathlib import Path
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("notebook_builder_test", ROOT / "scripts/build_notebook.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_committed_notebook_matches_builder_and_has_no_claimed_outputs():
    notebook = nbformat.read(ROOT / "notebooks/cs114_safeview.ipynb", as_version=4)
    nbformat.validate(notebook)
    expected = builder.build()
    assert [(c.cell_type, c.source) for c in notebook.cells] == [(c.cell_type, c.source) for c in expected.cells]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "notebook", "exec")
            assert cell.execution_count is None and cell.outputs == []


def test_readonly_execution_disables_even_previously_enabled_flags(tmp_path, monkeypatch):
    import IPython.display
    import safeview_ml.provenance
    import safeview_ml.remote_data
    import safeview_ml.training
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only notebook attempted network/setup/training/test")

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("SAFEVIEW_NOTEBOOK_READ_ONLY", "1")
    monkeypatch.setenv("SAFEVIEW_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("SAFEVIEW_RUN_ID", "")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(safeview_ml.remote_data, "load_github_dataset", forbidden)
    monkeypatch.setattr(safeview_ml.training, "train_experiment", forbidden)
    monkeypatch.setattr(safeview_ml.training, "evaluate_locked", forbidden)
    monkeypatch.setattr(safeview_ml.provenance, "environment_metadata", lambda: {
        "python": "software-fixture", "platform": "offline", "versions": {},
    })
    monkeypatch.setattr(IPython.display, "display", lambda *args, **kwargs: None)
    notebook = builder.build()
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
