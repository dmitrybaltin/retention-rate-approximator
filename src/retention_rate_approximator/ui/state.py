from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from retention_rate_approximator.core.plotting import YAxisMode, plot_dataset_preview

SOURCE_NONE = 'none'
SOURCE_UPLOADED = 'uploaded'
SOURCE_GENERATED = 'generated'


def build_source_status(source_kind: str, row_count: int | None = None, file_path: str | None = None) -> str:
    if source_kind in {SOURCE_UPLOADED, SOURCE_GENERATED} and file_path:
        file_name = Path(file_path).name
        if row_count is not None:
            return f'**Selected source file:** {file_name} ({row_count} rows)'
        return f'**Selected source file:** {file_name}'
    return '**Selected dataset:** none'


def build_preview_plot_update(frame: pd.DataFrame | None, y_axis_mode: YAxisMode) -> object:
    if frame is None or frame.empty:
        return gr.update(value=None, visible=False)
    return gr.update(value=plot_dataset_preview(frame, y_axis_mode), visible=True)


def build_preview_table_update(frame: pd.DataFrame | None) -> object:
    if frame is None or frame.empty:
        return gr.update(value=None, visible=False)
    return gr.update(value=frame, visible=True)


def fit_source_ui_updates(
    source_kind: str,
    preview_frame: pd.DataFrame | None,
    file_path: str | None = None,
    y_axis_mode: YAxisMode = 'zero',
) -> tuple[str, object, object, object, object, object]:
    row_count = None if preview_frame is None else len(preview_frame)
    return (
        build_source_status(source_kind, row_count, file_path),
        build_preview_plot_update(preview_frame, y_axis_mode),
        build_preview_table_update(preview_frame),
        gr.update(interactive=source_kind != SOURCE_NONE),
        gr.update(value=file_path, visible=True),
        gr.update(visible=source_kind == SOURCE_GENERATED),
    )
