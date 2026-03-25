from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import pandas as pd

from retention_rate_approximator.data import load_retention_csv, load_retention_frame, save_retention_csv
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
CUSTOM_CSS: Final[str] = """
#fit-pane .gr-block,
#demo-pane .gr-block {
  min-height: 0;
}
#fit-pane .pane-column,
#demo-pane .pane-column {
  gap: 0.6rem;
}
#fit-pane .pane-header,
#demo-pane .pane-header {
  align-items: start;
  margin-bottom: 0.25rem;
}
#fit-pane .pane-header .gr-markdown,
#demo-pane .pane-header .gr-markdown {
  margin: 0;
}
#fit-pane .pane-header .gr-button,
#fit-pane .pane-header .gr-downloadbutton,
#demo-pane .pane-header .gr-button,
#demo-pane .pane-header .gr-downloadbutton {
  margin-top: 0;
}
#fit-pane .pane-grid-row,
#demo-pane .pane-grid-row {
  gap: 0.75rem;
  margin-top: 0;
  margin-bottom: 0;
}
#fit-pane .pane-grid-row .gr-block,
#fit-pane .pane-column .gr-block,
#demo-pane .pane-grid-row .gr-block,
#demo-pane .pane-column .gr-block {
  margin-top: 0;
}
#fit-pane .pane-summary,
#demo-pane .pane-summary {
  margin-top: 0.25rem;
}
"""
TOOLTIP_JS: Final[str] = """
() => {
  const applyTooltips = () => {
    const tooltipMap = {
      'demo-download-button': 'Download the generated dataset as a CSV file.',
      'demo-push-button': 'Send the generated dataset directly to the approximator without using a file.',
    };
    Object.entries(tooltipMap).forEach(([id, title]) => {
      const root = document.getElementById(id);
      if (!root) return;
      root.title = title;
      const button = root.querySelector('button');
      if (button) {
        button.title = title;
      }
    });
  };
  applyTooltips();
  const observer = new MutationObserver(applyTooltips);
  observer.observe(document.body, { childList: true, subtree: true });
}
"""


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


def _create_generated_frame(
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
) -> tuple[pd.DataFrame, object, str, str]:
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
    frame = pd.DataFrame(
        {
            'day_number': generated.day_numbers.detach().cpu().numpy(),
            'installs': generated.installs.detach().cpu().numpy(),
            'retention': generated.retention.detach().cpu().numpy(),
            'retention_mean': generated.retention_trend.detach().cpu().numpy(),
        }
    )
    figure = plot_synthetic_dataset(generated)
    details = (
        '### Demo dataset generated\n'
        f'- Days: {total_days}\n'
        f"- Patch dates: {', '.join(str(value) for value in generated.patches_dates) or 'none'}\n"
        f'- Anomaly days: {len(generated.bad_days)}'
    )
    return frame, figure, details, str(output_path)


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
) -> tuple[pd.DataFrame, object, str, str, pd.DataFrame]:
    frame, figure, details, output_path = _create_generated_frame(
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
    )
    return frame, figure, details, output_path, frame


def _fit_source_present(csv_file: str | None, generated_frame: pd.DataFrame | None, fit_has_data: bool) -> bool:
    return fit_has_data or csv_file is not None or (generated_frame is not None and not generated_frame.empty)


def use_generated_dataset_in_fit(generated_frame: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    if generated_frame is None or generated_frame.empty:
        raise gr.Error('Generate a demo dataset first.')
    return generated_frame, 'Generated dataset is ready for fitting. Go to the Fit CSV tab and click Fit model.'


def request_generated_dataset_transfer(
    generated_frame: pd.DataFrame | None,
    csv_file: str | None,
    fit_has_data: bool,
) -> tuple[object, str, object, object, bool]:
    if generated_frame is None or generated_frame.empty:
        raise gr.Error('Generate a demo dataset first.')
    if _fit_source_present(csv_file, None, fit_has_data):
        warning = 'Approximator already has data. Pushing the generated dataset will overwrite it. Click confirm to continue.'
        return (
            gr.update(),
            warning,
            gr.update(visible=True),
            gr.update(visible=False),
            fit_has_data,
        )
    return (
        generated_frame,
        'Generated dataset is ready for fitting. Go to the Fit CSV tab and click Fit model.',
        gr.update(visible=False),
        gr.update(visible=True),
        True,
    )


def confirm_generated_dataset_transfer(generated_frame: pd.DataFrame | None) -> tuple[pd.DataFrame, str, object, object, bool]:
    if generated_frame is None or generated_frame.empty:
        raise gr.Error('Generate a demo dataset first.')
    return (
        generated_frame,
        'Generated dataset replaced the previous approximator input. Go to the Fit CSV tab and click Fit model.',
        gr.update(visible=False),
        gr.update(visible=True),
        True,
    )


def on_csv_selected(csv_file: str | None) -> tuple[str, bool, gr.update]:
    if csv_file is None:
        return 'Source: upload a CSV or send a generated dataset from the demo tab.', False, gr.update(visible=False)
    return 'Source: uploaded CSV selected. Click Fit model to use it.', True, gr.update(visible=False)


def fit_uploaded_dataset(
    csv_file: str | None,
    generated_frame: pd.DataFrame | None,
    first_day_of_week: int,
    main_function_type: str,
    chain_function_type: str,
    connector_type: str,
    patches_dates: str,
    bad_dates: str,
    week_function_weights: str,
    training_mode: str,
    exclude_patch_dates: bool,
) -> tuple[object, pd.DataFrame, str, str, bool]:
    if csv_file is not None:
        dataset = load_retention_csv(csv_file)
        source_label = 'uploaded CSV'
    elif generated_frame is not None and not generated_frame.empty:
        dataset = load_retention_frame(generated_frame)
        source_label = 'generated dataset'
    else:
        raise gr.Error('Upload a CSV file or send a generated dataset to the approximator first.')

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
    summary = f'Source: {source_label}\n\n' + build_training_summary(result)
    output_path = _build_dataset_download_path('fit_predictions.csv')
    frame.to_csv(output_path, index=False)
    return figure, frame, summary, str(output_path), True


def build_app() -> gr.Blocks:
    main_function_choices = [spec.name for spec in ApproximatorsFactory.main_functions]
    chain_function_choices = [spec.name for spec in ApproximatorsFactory.chain_functions]
    connector_choices = [connector[0] for connector in ApproximatorsFactory.connectors]

    with gr.Blocks(title='Retention Rate Approximator', css=CUSTOM_CSS, js=TOOLTIP_JS) as app:
        generated_state = gr.State(value=None)
        fit_has_data_state = gr.State(value=False)

        gr.Markdown(
            """
            # Retention Rate Approximator

            Upload retention CSV data or generate a demo dataset, then fit the original notebook model through a simpler web UI.
            Expected CSV columns: `date` or `day_number`, `installs`, `retention`, `retention_mean`.
            """
        )

        with gr.Tab('Fit CSV'):
            with gr.Row(elem_id='fit-pane'):
                with gr.Column(scale=4, elem_classes='pane-column'):
                    with gr.Row(elem_classes='pane-header'):
                        gr.Markdown('### Configure and Fit')
                        fit_button = gr.Button('Fit', variant='primary', size='sm')
                    fit_source_status = gr.Markdown('Source: upload a CSV or send a generated dataset from the demo tab.', elem_classes='pane-summary')
                    csv_file = gr.File(label='Retention CSV', file_types=['.csv'], type='filepath')
                    generated_preview = gr.Dataframe(label='Generated dataset passed from demo tab', interactive=False, visible=False)
                    with gr.Row(elem_classes='pane-grid-row'):
                        first_day_of_week = gr.Slider(label='First day of week', minimum=0, maximum=6, value=0, step=1)
                        training_mode = gr.Radio(label='Training preset', choices=['Fast', 'Standard'], value='Standard')
                    with gr.Row(elem_classes='pane-grid-row'):
                        main_function_type = gr.Dropdown(label='Main function', choices=main_function_choices, value=main_function_choices[4])
                        chain_function_type = gr.Dropdown(label='Patch function', choices=chain_function_choices, value=chain_function_choices[0])
                    with gr.Row(elem_classes='pane-grid-row'):
                        connector_type = gr.Dropdown(label='Connector', choices=connector_choices, value=connector_choices[0])
                        exclude_patch_dates = gr.Checkbox(label='Exclude patch dates from training', value=True)
                    with gr.Row(elem_classes='pane-grid-row'):
                        patches_dates = gr.Textbox(label='Patch dates', value='', placeholder='30, 60, 90')
                        bad_dates = gr.Textbox(label='Bad dates', value='', placeholder='12, 45')
                    week_function_weights = gr.Textbox(label='Week weights', value='1, 1, 1, 1, 1, 1, 1')
                with gr.Column(scale=6, elem_classes='pane-column'):
                    with gr.Row(elem_classes='pane-header'):
                        gr.Markdown('### Fit results')
                        predictions_download = gr.File(label='Predictions CSV')
                    fit_plot = gr.Plot(label='Fit plot')
                    fit_table = gr.Dataframe(label='Predictions', interactive=False)
                    fit_summary = gr.Markdown(elem_classes='pane-summary')

            csv_file.change(
                fn=on_csv_selected,
                inputs=[csv_file],
                outputs=[fit_source_status, fit_has_data_state, generated_preview],
            )

            fit_button.click(
                fn=fit_uploaded_dataset,
                inputs=[
                    csv_file,
                    generated_state,
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
                outputs=[fit_plot, fit_table, fit_summary, predictions_download, fit_has_data_state],
            )

        with gr.Tab('Generate demo'):
            with gr.Row(elem_id='demo-pane'):
                with gr.Column(scale=4, elem_classes='pane-column'):
                    with gr.Row(elem_classes='pane-header'):
                        gr.Markdown('### Fill settings and Generate')
                        demo_button = gr.Button('Generate', variant='primary', size='sm')
                    with gr.Row(elem_classes='pane-grid-row'):
                        demo_total_days = gr.Slider(label='Days', minimum=30, maximum=365, value=160, step=1)
                        demo_first_day_of_week = gr.Slider(label='First day of week', minimum=0, maximum=6, value=2, step=1)
                    with gr.Row(elem_classes='pane-grid-row'):
                        demo_daily_installs_mean = gr.Slider(label='Mean installs', minimum=100, maximum=10000, value=1000, step=50)
                        demo_daily_installs_sigma = gr.Slider(label='Install sigma', minimum=10, maximum=2000, value=200, step=10)
                    with gr.Row(elem_classes='pane-grid-row'):
                        demo_main_function_type = gr.Dropdown(label='Main function', choices=main_function_choices, value=main_function_choices[4])
                        demo_chain_function_type = gr.Dropdown(label='Patch function', choices=chain_function_choices, value=chain_function_choices[0])
                    with gr.Row(elem_classes='pane-grid-row'):
                        demo_main_function_weights = gr.Textbox(label='Main function weights', value='0.5, 0.4, 0.05')
                        demo_chain_function_weights = gr.Textbox(label='Patch weights', value='0.01, 0.02, 0.02, 0.03, 0.04')
                    with gr.Row(elem_classes='pane-grid-row'):
                        demo_patches_dates = gr.Textbox(label='Patch dates', value='30, 60, 90, 120, 150')
                        demo_week_function_weights = gr.Textbox(label='Week weights', value='1, 1, 1, 1, 1.05, 1.05, 0.9')
                    demo_summary = gr.Markdown(elem_classes='pane-summary')
                with gr.Column(scale=6, elem_classes='pane-column'):
                    with gr.Row(elem_classes='pane-header'):
                        gr.Markdown('### Generated dataset')
                        demo_download_button = gr.DownloadButton('Download', elem_id='demo-download-button', visible=False, size='sm')
                        send_to_fit_button = gr.Button('Puch to approximator', elem_id='demo-push-button', visible=False, size='sm')
                    demo_plot = gr.Plot(label='Synthetic dataset')
                    generated_table = gr.Dataframe(label='Generated data', interactive=False)
                    overwrite_warning = gr.Markdown(visible=False)
                    confirm_overwrite_button = gr.Button('Confirm overwrite', variant='secondary', size='sm', visible=False)

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
                outputs=[generated_state, demo_plot, demo_summary, demo_download_button, generated_table],
            ).then(
                fn=lambda: (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
                inputs=None,
                outputs=[demo_download_button, send_to_fit_button, overwrite_warning, confirm_overwrite_button],
            )

            send_to_fit_button.click(
                fn=request_generated_dataset_transfer,
                inputs=[generated_state, csv_file, fit_has_data_state],
                outputs=[generated_preview, fit_source_status, overwrite_warning, confirm_overwrite_button, fit_has_data_state],
            ).then(
                fn=lambda frame: gr.update(visible=frame is not None),
                inputs=[generated_preview],
                outputs=[generated_preview],
            )

            confirm_overwrite_button.click(
                fn=confirm_generated_dataset_transfer,
                inputs=[generated_state],
                outputs=[generated_preview, fit_source_status, overwrite_warning, confirm_overwrite_button, fit_has_data_state],
            ).then(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=[generated_preview],
            )

    return app


app = build_app()


if __name__ == '__main__':
    app.launch()