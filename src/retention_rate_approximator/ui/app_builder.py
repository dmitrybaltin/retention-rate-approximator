from __future__ import annotations

import inspect
import os
from typing import Final

import gradio as gr

from retention_rate_approximator.core.modeling import ApproximatorsFactory
from retention_rate_approximator.ui.handlers import (
    clear_generated_source,
    confirm_generated_dataset_transfer,
    fit_uploaded_dataset,
    generate_demo_dataset,
    on_csv_selected,
    request_generated_dataset_transfer,
)
from retention_rate_approximator.ui.state import SOURCE_NONE, build_source_status

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
#fit-pane .gr-tabitem,
#demo-pane .gr-tabitem {
  padding-top: 0.5rem;
}
"""


def _supports_parameter(callable_obj: object, parameter_name: str) -> bool:
    return parameter_name in inspect.signature(callable_obj).parameters


BLOCKS_SUPPORTS_CSS: Final[bool] = _supports_parameter(gr.Blocks, 'css')
LAUNCH_SUPPORTS_CSS: Final[bool] = _supports_parameter(gr.Blocks.launch, 'css')


def create_launch_kwargs() -> dict[str, object]:
    launch_kwargs: dict[str, object] = {
        'server_name': '0.0.0.0',
        'server_port': int(os.environ.get('PORT', '7860')),
        'ssr_mode': False,
    }
    if LAUNCH_SUPPORTS_CSS:
        launch_kwargs['css'] = CUSTOM_CSS
    return launch_kwargs


def build_app() -> gr.Blocks:
    main_function_choices = [spec.name for spec in ApproximatorsFactory.main_functions]
    chain_function_choices = [spec.name for spec in ApproximatorsFactory.chain_functions]
    connector_choices = [connector[0] for connector in ApproximatorsFactory.connectors]
    blocks_kwargs: dict[str, object] = {'title': 'Retention Rate Approximator'}
    if BLOCKS_SUPPORTS_CSS:
        blocks_kwargs['css'] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as app:
        generated_csv_state = gr.State(value=None)
        fit_source_kind_state = gr.State(value=SOURCE_NONE)
        session_id_state = gr.State(value=None)

        gr.Markdown(
            """
            # Retention Rate Approximator

            Upload retention CSV data or generate a demo dataset, then fit the original notebook model through a simpler web UI.
            Expected CSV columns: `date` or `day_number`, `installs`, `retention`, `retention_mean`.
            """
        )
        with gr.Tab('Fit CSV'):
            with gr.Row(elem_id='fit-pane'):
                with gr.Column(scale=1, elem_classes='pane-column'):
                    with gr.Row(elem_classes='pane-header'):
                        gr.Markdown('### Configure and Fit')
                        fit_button = gr.Button('Fit', variant='primary', size='sm', interactive=False)
                    gr.Markdown('#### Data source')
                    fit_source_status = gr.Markdown(build_source_status(SOURCE_NONE), elem_classes='pane-summary')
                    csv_file = gr.File(label='Retention CSV', file_types=['.csv'], type='filepath')
                    clear_generated_source_button = gr.Button('Clear generated dataset', size='sm', visible=False)
                    with gr.Tab('Chart'):
                        source_preview_plot = gr.Plot(visible=False)
                    with gr.Tab('Table'):
                        generated_preview = gr.Dataframe(interactive=False, visible=False)
                    gr.Markdown('#### Model setup')
                    with gr.Row(elem_classes='pane-grid-row'):
                        main_function_type = gr.Dropdown(label='Main function', choices=main_function_choices, value=main_function_choices[4])
                        chain_function_type = gr.Dropdown(label='Patch function', choices=chain_function_choices, value=chain_function_choices[0])
                        connector_type = gr.Dropdown(label='Connector', choices=connector_choices, value=connector_choices[0])
                    gr.Markdown('#### Training and filters')
                    training_mode = gr.Radio(label='Training preset', choices=['Fast', 'Standard'], value='Standard')
                    with gr.Row(elem_classes='pane-grid-row'):
                        first_day_of_week = gr.Slider(label='First day of week', minimum=0, maximum=6, value=0, step=1)
                        week_function_weights = gr.Textbox(label='Week weights', value='1, 1, 1, 1, 1, 1, 1')
                    with gr.Row(elem_classes='pane-grid-row'):
                        patches_dates = gr.Textbox(label='Patch dates', value='', placeholder='30, 60, 90')
                        exclude_patch_dates = gr.Checkbox(label='Exclude patch dates from training', value=True)
                    bad_dates = gr.Textbox(label='Bad dates', value='', placeholder='12, 45')
                with gr.Column(scale=1, elem_classes='pane-column'):
                    with gr.Row(elem_classes='pane-header'):
                        gr.Markdown('### Results')
                        predictions_download = gr.DownloadButton('Download Results', interactive=False, size='sm')
                    with gr.Tab('Chart'):
                        fit_plot = gr.Plot()
                    with gr.Tab('Table'):
                        fit_table = gr.Dataframe(interactive=False)
                    gr.Markdown('### Run summary')
                    fit_summary = gr.Markdown(elem_classes='pane-summary')

            csv_file.change(
                fn=on_csv_selected,
                inputs=[csv_file],
                outputs=[fit_source_status, source_preview_plot, generated_preview, fit_button, clear_generated_source_button, fit_source_kind_state],
            )

            clear_generated_source_button.click(
                fn=clear_generated_source,
                inputs=None,
                outputs=[fit_source_status, source_preview_plot, generated_preview, fit_button, csv_file, clear_generated_source_button, fit_source_kind_state],
            )

            fit_button.click(
                fn=fit_uploaded_dataset,
                inputs=[
                    csv_file,
                    fit_source_kind_state,
                    first_day_of_week,
                    main_function_type,
                    chain_function_type,
                    connector_type,
                    patches_dates,
                    bad_dates,
                    week_function_weights,
                    training_mode,
                    exclude_patch_dates,
                    session_id_state,
                ],
                outputs=[fit_plot, fit_table, fit_summary, predictions_download, fit_source_kind_state, session_id_state],
            )

        with gr.Tab('Generate demo'):
            with gr.Row(elem_id='demo-pane'):
                with gr.Column(scale=1, elem_classes='pane-column'):
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
                with gr.Column(scale=1, elem_classes='pane-column'):
                    with gr.Row(elem_classes='pane-header'):
                        gr.Markdown('### Generated dataset')
                        demo_download_button = gr.DownloadButton('Download', visible=False, size='sm')
                        send_to_fit_button = gr.Button('Puch to approximator', visible=False, size='sm')
                    with gr.Tab('Chart'):
                        demo_plot = gr.Plot()
                    with gr.Tab('Table'):
                        generated_table = gr.Dataframe(interactive=False)
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
                    session_id_state,
                ],
                outputs=[generated_csv_state, demo_plot, demo_summary, demo_download_button, generated_table, session_id_state],
            ).then(
                fn=lambda: (
                    gr.update(visible=True),
                    gr.update(value='Puch to approximator', visible=True, interactive=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                ),
                inputs=None,
                outputs=[demo_download_button, send_to_fit_button, overwrite_warning, confirm_overwrite_button],
            )

            send_to_fit_button.click(
                fn=request_generated_dataset_transfer,
                inputs=[generated_csv_state, fit_source_kind_state],
                outputs=[fit_source_status, source_preview_plot, generated_preview, overwrite_warning, confirm_overwrite_button, fit_button, csv_file, clear_generated_source_button, fit_source_kind_state],
            )

            confirm_overwrite_button.click(
                fn=confirm_generated_dataset_transfer,
                inputs=[generated_csv_state],
                outputs=[fit_source_status, source_preview_plot, generated_preview, overwrite_warning, confirm_overwrite_button, fit_button, csv_file, clear_generated_source_button, fit_source_kind_state],
            )

    return app


app = build_app()

