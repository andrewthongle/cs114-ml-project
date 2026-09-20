"""Execute the existing notebook without replacing its explanations or cells."""
from pathlib import Path
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/cs114_safeview.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
NotebookClient(notebook, km=manager, timeout=3600,
               resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(notebook, path)
print(f"Executed and saved: {path}")
