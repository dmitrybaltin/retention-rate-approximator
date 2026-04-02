from __future__ import annotations

import os
import unittest

import pandas as pd

from app import (
    SOURCE_GENERATED,
    SOURCE_NONE,
    SOURCE_UPLOADED,
    _build_session_download_path,
    clear_generated_source,
    confirm_generated_dataset_transfer,
    fit_uploaded_dataset,
    on_csv_selected,
    request_generated_dataset_transfer,
    use_generated_dataset_in_fit,
)


class AppTests(unittest.TestCase):
    def test_download_path_preserves_csv_extension(self) -> None:
        path, session_id = _build_session_download_path('demo_dataset.csv', 'test-session')
        self.assertEqual(path.name, 'demo_dataset.csv')
        self.assertEqual(path.suffix, '.csv')
        self.assertEqual(path.parent.name, 'test-session')
        self.assertEqual(session_id, 'test-session')

    def test_use_generated_dataset_returns_path_and_status(self) -> None:
        returned_path, status = use_generated_dataset_in_fit('.artifacts/test-session/demo_dataset.csv')
        self.assertTrue(returned_path.endswith('demo_dataset.csv'))
        self.assertIn('ready for fitting', status)

    def test_request_generated_dataset_transfer_requires_confirmation_when_source_exists(self) -> None:
        status, preview_plot, preview_table, warning, confirm_button, fit_button, csv_file, clear_button, source_kind, preview_frame = request_generated_dataset_transfer(
            '.artifacts/test-session/demo_dataset.csv',
            SOURCE_UPLOADED,
        )
        self.assertNotIn('value', status)
        self.assertNotIn('value', preview_plot)
        self.assertNotIn('value', preview_table)
        self.assertIn('overwrite', warning)
        self.assertTrue(confirm_button['visible'])
        self.assertNotIn('interactive', fit_button)
        self.assertNotIn('value', csv_file)
        self.assertNotIn('visible', clear_button)
        self.assertEqual(source_kind, SOURCE_UPLOADED)
        self.assertIsNone(preview_frame)

    def test_confirm_generated_dataset_transfer_selects_preview(self) -> None:
        os.makedirs('.artifacts/test-session', exist_ok=True)
        csv_file = '.artifacts/test-session/demo_dataset.csv'
        frame = pd.DataFrame(
            {
                'day_number': [0.0, 1.0],
                'installs': [100.0, 120.0],
                'retention': [0.4, 0.35],
                'retention_mean': [0.41, 0.36],
            }
        )
        frame.to_csv(csv_file, index=False)
        try:
            status, preview_plot, preview_table, warning, confirm_button, fit_button, csv_update, clear_button, source_kind, preview_frame = confirm_generated_dataset_transfer(csv_file)
            self.assertIn('demo_dataset.csv', status)
            self.assertTrue(preview_plot['visible'])
            self.assertEqual(len(preview_table['value']), 2)
            self.assertFalse(warning['visible'])
            self.assertFalse(confirm_button['visible'])
            self.assertTrue(fit_button['interactive'])
            self.assertEqual(csv_update['value'], csv_file)
            self.assertTrue(clear_button['visible'])
            self.assertEqual(source_kind, SOURCE_GENERATED)
            self.assertEqual(len(preview_frame), 2)
        finally:
            if os.path.exists(csv_file):
                os.remove(csv_file)

    def test_clear_generated_source_resets_fit_input(self) -> None:
        status, preview_plot, preview_table, fit_button, csv_file, clear_button, source_kind, preview_frame = clear_generated_source()
        self.assertIn('none', status)
        self.assertFalse(preview_plot['visible'])
        self.assertFalse(preview_table['visible'])
        self.assertFalse(fit_button['interactive'])
        self.assertIsNone(csv_file['value'])
        self.assertFalse(clear_button['visible'])
        self.assertEqual(source_kind, SOURCE_NONE)
        self.assertIsNone(preview_frame)

    def test_on_csv_selected_shows_preview(self) -> None:
        os.makedirs('.artifacts/test-session', exist_ok=True)
        csv_file = '.artifacts/test-session/upload.csv'
        frame = pd.DataFrame(
            {
                'day_number': [0.0, 1.0],
                'installs': [100.0, 120.0],
                'retention': [0.4, 0.35],
                'retention_mean': [0.41, 0.36],
            }
        )
        frame.to_csv(csv_file, index=False)
        try:
            status, preview_plot, preview_table, fit_button, clear_button, source_kind, preview_frame = on_csv_selected(csv_file)
            self.assertIn('upload.csv', status)
            self.assertTrue(preview_plot['visible'])
            self.assertEqual(len(preview_table['value']), 2)
            self.assertTrue(fit_button['interactive'])
            self.assertFalse(clear_button['visible'])
            self.assertEqual(source_kind, SOURCE_UPLOADED)
            self.assertEqual(len(preview_frame), 2)
        finally:
            if os.path.exists(csv_file):
                os.remove(csv_file)

    def test_fit_uploaded_dataset_accepts_csv_file(self) -> None:
        os.makedirs('.artifacts/test-session', exist_ok=True)
        csv_file = '.artifacts/test-session/demo_dataset.csv'
        frame = pd.DataFrame(
            {
                'day_number': [0.0, 1.0, 2.0, 3.0],
                'installs': [1000.0, 1000.0, 1000.0, 1000.0],
                'retention': [0.45, 0.4, 0.35, 0.3],
                'retention_mean': [0.44, 0.39, 0.34, 0.29],
            }
        )
        frame.to_csv(csv_file, index=False)
        try:
            figure, predictions, summary, download_update, fit_source_kind, session_id, prediction_frame = fit_uploaded_dataset(
                csv_file=csv_file,
                fit_source_kind=SOURCE_GENERATED,
                first_day_of_week=0,
                main_function_type='w0',
                chain_function_type='w0',
                connector_type='mul',
                patches_dates='',
                bad_dates='',
                week_function_weights='1, 1, 1, 1, 1, 1, 1',
                training_mode='Fast',
                exclude_patch_dates=True,
                session_id='test-session',
            )
            self.assertIsNotNone(figure)
            self.assertEqual(len(predictions), 4)
            self.assertIn('Source: generated dataset', summary)
            self.assertEqual(download_update['value'].split('\\')[-1], 'fit_predictions.csv')
            self.assertIn('test-session', download_update['value'])
            self.assertTrue(download_update['interactive'])
            self.assertEqual(fit_source_kind, SOURCE_GENERATED)
            self.assertEqual(session_id, 'test-session')
            self.assertEqual(len(prediction_frame), 4)
        finally:
            if os.path.exists(csv_file):
                os.remove(csv_file)


if __name__ == '__main__':
    unittest.main()
