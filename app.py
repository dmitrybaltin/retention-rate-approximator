from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retention_rate_approximator.ui.app_builder import app, build_app, create_launch_kwargs
from retention_rate_approximator.ui.handlers import (
    confirm_generated_dataset_transfer,
    fit_uploaded_dataset,
    generate_demo_dataset,
    on_csv_selected,
    request_generated_dataset_transfer,
    use_generated_dataset_in_fit,
)
from retention_rate_approximator.ui.state import SOURCE_GENERATED, SOURCE_NONE, SOURCE_UPLOADED
from retention_rate_approximator.core.app_support import build_session_download_path as _build_session_download_path

if __name__ == '__main__':
    app.launch(**create_launch_kwargs())
