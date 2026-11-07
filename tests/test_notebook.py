"""The walkthrough notebook is committed WITHOUT outputs: outputs show text of the document it ran on,
which may be a copyrighted course document (Kernel -> Restart & Clear All Outputs before committing)."""
import json
from pathlib import Path

NOTEBOOKS = sorted(Path(__file__).parent.parent.glob("notebooks/*.ipynb"))


def test_notebooks_have_no_outputs():
    for nb_path in NOTEBOOKS:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        dirty = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and c.get("outputs")]
        assert not dirty, f"{nb_path.name}: clear the outputs of cells {dirty} before committing"


def test_notebook_code_compiles():
    for nb_path in NOTEBOOKS:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"{nb_path.name}[cell {i}]", "exec")
