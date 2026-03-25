from __future__ import annotations

from pathlib import Path
from typing import Final

import gradio as gr
import pandas as pd

from retention_rate_approximator.data import load_retention_csv, save_retention_csv
from retention_rate_approximator.modeling import ApproximatorsFactory
from retention_rate_approximator.plotting import (
    FitPlotData,
    build_prediction_frame,
    build_training_summary,
    plot_fit_results,
    plot_synthetic_dataset,
)
from retention_rate_approximator.synthetic import generate_retention_dataset
from retention_rate_approximator.training import TrainingPhase, train_retention_model

ARTIFACTS_DIR: Final[Path] = Path('.artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _parse_int_list(raw_value: str) -> list[int]:
    if not raw_value.strip():
        return []
    return [int(item.strip()) for item in raw_value.split(',') if item.strip()]


def _parse_float_list(raw_value: str) -> list[float]:
    if not raw_value.strip():
        return []
    return [float(item.strip()) for item in raw_value.split(',') if item.strip()]


def _resolve_training_strategy(mode: str) -> list[TrainingPhase]:
    if mode == 'Fast':
        return [
            TrainingPhase('Adam', 150, True, False, 0.01),
            TrainingPhase('Adam', 150, True, True, 0.005),
        ]
    return [
        TrainingPhase('Adam', 300, True, False, 0.01),
        TrainingPhase('LBFGS', 3, True, False, 0.01),
        TrainingPhase('Adam', 300, True, True, 0.001),
    ]


def _build_dataset_download_path(name: str) -> Path:
    source = Path(name)
    safe_stem = ''.join(char if char.isalnum() or char in '-_' else '_' for char in source.stem)
    suffix = source.suffix or '.csv'
    return ARTIFACTS_DIR / f'{safe_stem}{suffix}'


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
) -> tuple[str, object, str]:
    patch_values = _parse_int_list(patches_dates)
    main_weight_values = _parse_float_list(main_function_weights)
    chain_weight_values = _parse_float_list(chain_function_weights)
    week_weight_values = _parse_float_list(week_function_weights)
    generated = generate_retention_dataset(
        total_days=total_days,
        first_day_of_week=first_day_of_week,
        patches_dates=patch_values,
        main_function_type=main_function_type,
        chains_functions_type=chain_function_type,
        main_function_weights=main_weight_values,
        chains_functions_weights=chain_weight_values,
        week_function_weights=week_weight_values,
        daily_installs_mean=daily_installs_mean,
        daily_installs_sigma=daily_installs_sigma,
    )
    output_path = _build_dataset_download_path('demo_dataset.csv')
    save_retention_csv(
        path=output_path,
        day_numbers=generated.day_numbers,
        installs=generated.installs,
        retention=generated.retention,
        retention_mean=generated.retention_trend,
    )
    figure = plot_synthetic_dataset(generated)
    details = (
        '### Demo dataset generated\n'
        f'- Days: {total_days}\n'
        f"- Patch dates: {', '.join(str(value) for value in generated.patches_dates) or 'none'}\n"
        f'- Anomaly days: {len(generated.bad_days)}'
    )
    return str(output_path), figure, details


def fit_uploaded_dataset(
    csv_file: str | None,
    first_day_of_week: int,
    main_function_type: str,
    chain_function_type: str,
    connector_type: str,
    patches_dates: str,
    bad_dates: str,
    week_function_weights: str,
    training_mode: str,
    exclude_patch_dates: bool,
) -> tuple[object, pd.DataFrame, str, str]:
    if csv_file is None:
        raise gr.Error('Upload a CSV file first.')

    dataset = load_retention_csv(csv_file)
    patch_values = _parse_int_list(patches_dates)
    bad_date_values = _parse_int_list(bad_dates)
    week_weight_values = _parse_float_list(week_function_weights) if week_function_weights.strip() else None
    result = train_retention_model(
        dataset=dataset,
        first_day_of_week=first_day_of_week,
        main_function_type=main_function_type,
        chain_function_type=chain_function_type,
        connector_type=connector_type,
        patches_dates=patch_values,
        bad_dates=bad_date_values,
        exclude_patch_dates=exclude_patch_dates,
        week_function_initial_weights=week_weight_values,
        training_strategy=_resolve_training_strategy(training_mode),
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
    summary = build_training_summary(result)
    output_path = _build_dataset_download_path('fit_predictions.csv')
    frame.to_csv(output_path, index=False)
    return figure, frame, summary, str(output_path)


def build_app() -> gr.Blocks:
    main_function_choices = [spec.name for spec in ApproximatorsFactory.main_functions]
    chain_function_choices = [spec.name for spec in ApproximatorsFactory.chain_functions]
    connector_choices = [connector[0] for connector in ApproximatorsFactory.connectors]

    with gr.Blocks(title='Retention Rate Approximator') as app:
        gr.Markdown(
            """
            # Retention Rate Approximator

            Upload retention CSV data or generate a demo dataset, then fit the original notebook model through a simpler web UI.
            Expected CSV columns: `date` or `day_number`, `installs`, `retention`, `retention_mean`.
            """
        )

        with gr.Tab('Fit CSV'):
            with gr.Row():
                csv_file = gr.File(label='Retention CSV', file_types=['.csv'], type='filepath')
                predictions_download = gr.File(label='Predictions CSV')
            with gr.Row():
                first_day_of_week = gr.Slider(label='First day of week', minimum=0, maximum=6, value=0, step=1)
                training_mode = gr.Radio(label='Training preset', choices=['Fast', 'Standard'], value='Standard')
                exclude_patch_dates = gr.Checkbox(label='Exclude patch dates from training', value=True)
            with gr.Row():
                main_function_type = gr.Dropdown(label='Main function', choices=main_function_choices, value=main_function_choices[4])
                chain_function_type = gr.Dropdown(label='Patch function', choices=chain_function_choices, value=chain_function_choices[0])
                connector_type = gr.Dropdown(label='Connector', choices=connector_choices, value=connector_choices[0])
            with gr.Row():
                patches_dates = gr.Textbox(label='Patch dates', value='', placeholder='30, 60, 90')
                bad_dates = gr.Textbox(label='Bad dates', value='', placeholder='12, 45')
                week_function_weights = gr.Textbox(label='Week weights', value='1, 1, 1, 1, 1, 1, 1')
            fit_button = gr.Button('Fit model', variant='primary')
            fit_plot = gr.Plot(label='Fit plot')
            fit_table = gr.Dataframe(label='Predictions', interactive=False)
            fit_summary = gr.Markdown()

            fit_button.click(
                fn=fit_uploaded_dataset,
                inputs=[
                    csv_file,
                    first_day_of_week,
                    main_function_type,
                    chain_function_type,
                    connector_type,
                    patches_dates,
                    bad_dates,
                    week_function_weights,
                    training_mode,
                    exclude_patch_dates,
                ],
                outputs=[fit_plot, fit_table, fit_summary, predictions_download],
            )

        with gr.Tab('Generate demo'):
            with gr.Row():
                demo_download = gr.File(label='Demo CSV')
                demo_plot = gr.Plot(label='Synthetic dataset')
            with gr.Row():
                demo_total_days = gr.Slider(label='Days', minimum=30, maximum=365, value=160, step=1)
                demo_first_day_of_week = gr.Slider(label='First day of week', minimum=0, maximum=6, value=2, step=1)
                demo_daily_installs_mean = gr.Slider(label='Mean installs', minimum=100, maximum=10000, value=1000, step=50)
                demo_daily_installs_sigma = gr.Slider(label='Install sigma', minimum=10, maximum=2000, value=200, step=10)
            with gr.Row():
                demo_patches_dates = gr.Textbox(label='Patch dates', value='30, 60, 90, 120, 150')
                demo_main_function_type = gr.Dropdown(label='Main function', choices=main_function_choices, value=main_function_choices[4])
                demo_chain_function_type = gr.Dropdown(label='Patch function', choices=chain_function_choices, value=chain_function_choices[0])
            with gr.Row():
                demo_main_function_weights = gr.Textbox(label='Main function weights', value='0.5, 0.4, 0.05')
                demo_chain_function_weights = gr.Textbox(label='Patch weights', value='0.01, 0.02, 0.02, 0.03, 0.04')
                demo_week_function_weights = gr.Textbox(label='Week weights', value='1, 1, 1, 1, 1.05, 1.05, 0.9')
            demo_button = gr.Button('Generate demo dataset')
            demo_summary = gr.Markdown()

            demo_button.click(
                fn=generate_demo_dataset,
                inputs=[
                    demo_total_days,
                    demo_first_day_of_week,
                    demo_patches_dates,
                    demo_main_function_type,
                    demo_chain_function_type,
                    demo_main_function_weights,
                    demo_chain_function_weights,
                    demo_week_function_weights,
                    demo_daily_installs_mean,
                    demo_daily_installs_sigma,
                ],
                outputs=[demo_download, demo_plot, demo_summary],
            )

    return app


app = build_app()


if __name__ == '__main__':
    app.launch()