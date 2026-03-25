from __future__ import annotations

from pathlib import Path

import gradio as gr

from retention_rate_approximator.core.app_support import (
    build_session_download_path,
    create_generated_frame,
    parse_float_list,
    parse_int_list,
    resolve_training_strategy,
)
from retention_rate_approximator.core.data import load_retention_csv
from retention_rate_approximator.core.plotting import FitPlotData, build_prediction_frame, build_training_summary, plot_fit_results
from retention_rate_approximator.core.training import train_retention_model
from retention_rate_approximator.ui.state import SOURCE_GENERATED, SOURCE_NONE, SOURCE_UPLOADED, fit_source_ui_updates


def use_generated_dataset_in_fit(generated_csv_path: str | None) -> tuple[str, str]:
    if generated_csv_path is None:
        raise gr.Error('Generate a demo dataset first.')
    return generated_csv_path, 'Generated dataset CSV is ready for fitting.'


def generate_demo_dataset(
    total_days: int,
    first_day_of_week: int,
    patches_dates: str,
    main_function_type: str,
    chain_function_type: str,
    main_function_weights: str,
    chain_function_weights: str,
    week_function_weights: str,
    daily_installs_mean: int,
    daily_installs_sigma: int,
    session_id: str | None,
) -> tuple[str, object, str, str, object, str]:
    frame, figure, details, output_path, active_session_id = create_generated_frame(
        total_days,
        first_day_of_week,
        patches_dates,
        main_function_type,
        chain_function_type,
        main_function_weights,
        chain_function_weights,
        week_function_weights,
        daily_installs_mean,
        daily_installs_sigma,
        session_id,
    )
    return str(output_path), figure, details, str(output_path), frame, active_session_id


def request_generated_dataset_transfer(
    generated_csv_path: str | None,
    fit_source_kind: str,
) -> tuple[object, object, object, str, object, object, object, object, str]:
    if generated_csv_path is None:
        raise gr.Error('Generate a demo dataset first.')
    if fit_source_kind != SOURCE_NONE:
        warning = 'Approximator already has data. Pushing the generated dataset will overwrite it. Click confirm to continue.'
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            warning,
            gr.update(visible=True),
            gr.update(),
            gr.update(),
            gr.update(),
            fit_source_kind,
        )
    dataset = load_retention_csv(generated_csv_path)
    status, preview_plot, preview_table, fit_button, csv_input, clear_button = fit_source_ui_updates(
        SOURCE_GENERATED,
        dataset.frame,
        generated_csv_path,
    )
    gr.Info('Generated dataset added to Fit.')
    return (
        status,
        preview_plot,
        preview_table,
        gr.update(visible=False),
        gr.update(visible=False),
        fit_button,
        csv_input,
        clear_button,
        SOURCE_GENERATED,
    )


def confirm_generated_dataset_transfer(generated_csv_path: str | None) -> tuple[object, object, object, str, object, object, object, object, str]:
    if generated_csv_path is None:
        raise gr.Error('Generate a demo dataset first.')
    dataset = load_retention_csv(generated_csv_path)
    status, preview_plot, preview_table, fit_button, csv_input, clear_button = fit_source_ui_updates(
        SOURCE_GENERATED,
        dataset.frame,
        generated_csv_path,
    )
    gr.Info('Generated dataset replaced the current Fit input.')
    return (
        status,
        preview_plot,
        preview_table,
        gr.update(visible=False),
        gr.update(visible=False),
        fit_button,
        csv_input,
        clear_button,
        SOURCE_GENERATED,
    )


def on_csv_selected(csv_file: str | None) -> tuple[str, object, object, object, object, str]:
    if csv_file is None:
        status, preview_plot, preview_table, fit_button, _, clear_button = fit_source_ui_updates(SOURCE_NONE, None)
        return status, preview_plot, preview_table, fit_button, clear_button, SOURCE_NONE
    dataset = load_retention_csv(csv_file)
    source_kind = SOURCE_GENERATED if Path(csv_file).name == 'demo_dataset.csv' else SOURCE_UPLOADED
    status, preview_plot, preview_table, fit_button, _, clear_button = fit_source_ui_updates(source_kind, dataset.frame, csv_file)
    return status, preview_plot, preview_table, fit_button, clear_button, source_kind


def clear_generated_source() -> tuple[str, object, object, object, object, object, str]:
    status, preview_plot, preview_table, fit_button, csv_input, clear_button = fit_source_ui_updates(SOURCE_NONE, None)
    return status, preview_plot, preview_table, fit_button, csv_input, clear_button, SOURCE_NONE


def fit_uploaded_dataset(
    csv_file: str | None,
    fit_source_kind: str,
    first_day_of_week: int,
    main_function_type: str,
    chain_function_type: str,
    connector_type: str,
    patches_dates: str,
    bad_dates: str,
    week_function_weights: str,
    training_mode: str,
    exclude_patch_dates: bool,
    session_id: str | None,
) -> tuple[object, object, str, object, str, str]:
    if csv_file is None:
        raise gr.Error('Upload a CSV file or send a generated dataset to the approximator first.')

    dataset = load_retention_csv(csv_file)
    source_kind = fit_source_kind if fit_source_kind != SOURCE_NONE else SOURCE_UPLOADED
    source_label = 'generated dataset' if source_kind == SOURCE_GENERATED else 'uploaded CSV'

    result = train_retention_model(
        dataset=dataset,
        first_day_of_week=first_day_of_week,
        main_function_type=main_function_type,
        chain_function_type=chain_function_type,
        connector_type=connector_type,
        patches_dates=parse_int_list(patches_dates),
        bad_dates=parse_int_list(bad_dates),
        exclude_patch_dates=exclude_patch_dates,
        week_function_initial_weights=parse_float_list(week_function_weights) if week_function_weights.strip() else None,
        training_strategy=resolve_training_strategy(training_mode),
    )
    figure = plot_fit_results(
        FitPlotData(
            day_numbers=dataset.day_numbers,
            retention=dataset.retention,
            predicted=result.predicted,
            predicted_trend=result.predicted_trend,
            used_day_numbers=result.used_day_numbers,
            used_retention=result.used_retention,
            retention_mean=dataset.retention_mean,
        )
    )
    frame = build_prediction_frame(dataset.day_numbers, dataset.retention, result.predicted, result.predicted_trend)
    summary = f'Source: {source_label}\n\n' + build_training_summary(result)
    output_path, active_session_id = build_session_download_path('fit_predictions.csv', session_id)
    frame.to_csv(output_path, index=False)
    return figure, frame, summary, gr.update(value=str(output_path), interactive=True), source_kind, active_session_id