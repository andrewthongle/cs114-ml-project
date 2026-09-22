"""Execute saved-result views; never turn a user's action flags into downloads/training."""
import os
from pathlib import Path
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/cs114_safeview.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # Refuse older notebooks without the explicit read-only guard rather than
    # assuming edited training/download flags are safe to execute automatically.
    tagged = [cell for cell in notebook.cells if "safeview-configuration" in cell.metadata.get("tags", [])]
    if len(tagged) != 1 or "SAFEVIEW_NOTEBOOK_READ_ONLY" not in tagged[0].source:
        raise ValueError("Notebook lacks read-only configuration guard; rebuild or execute it manually.")
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    env = {**os.environ, "SAFEVIEW_NOTEBOOK_READ_ONLY": "1"}
    NotebookClient(notebook, km=manager, timeout=600,
                   resources={"metadata": {"path": str(root)}}).execute(env=env)
    nbformat.write(notebook, path)
    print(f"Executed saved-results mode and saved: {path}")


if __name__ == "__main__":
    main()
