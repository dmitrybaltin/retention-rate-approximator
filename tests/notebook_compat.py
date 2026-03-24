from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any


def load_notebook_namespace() -> dict[str, Any]:
    notebook_path = Path(__file__).resolve().parents[1] / 'retention-rate-approximator.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    code_cells = [''.join(cell.get('source', [])) for cell in notebook['cells'] if cell.get('cell_type') == 'code']

    google_module = types.ModuleType('google')
    colab_module = types.ModuleType('google.colab')
    colab_module.files = types.SimpleNamespace(download=lambda *args, **kwargs: None)
    google_module.colab = colab_module
    sys.modules['google'] = google_module
    sys.modules['google.colab'] = colab_module

    namespace: dict[str, Any] = {}
    exec(code_cells[0], namespace, namespace)
    return namespace