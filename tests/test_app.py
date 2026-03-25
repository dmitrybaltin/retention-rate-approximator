from __future__ import annotations

import unittest

import pandas as pd

from app import (
    _build_dataset_download_path,
    confirm_generated_dataset_transfer,
    fit_uploaded_dataset,
    request_generated_dataset_transfer,
    use_generated_dataset_in_fit,
)


class AppTests(unittest.TestCase):
    def test_download_path_preserves_csv_extension(self) -> None:
        path = _build_dataset_download_path('demo_dataset.csv')
        self.assertTrue(path.name.startswith('demo_dataset-'))
        self.assertEqual(path.suffix, '.csv')

    def test_download_paths_are_unique(self) -> None:
        first_path = _build_dataset_download_path('demo_dataset.csv')
        second_path = _build_dataset_download_path('demo_dataset.csv')
        self.assertNotEqual(first_path, second_path)

    def test_use_generated_dataset_returns_frame_and_status(self) -> None:
        frame = pd.DataFrame(
            {
                'day_number': [0.0, 1.0, 2.0],
                'installs': [100.0, 120.0, 140.0],
                'retention': [0.4, 0.35, 0.3],
                'retention_mean': [0.41, 0.36, 0.31],
            }
        )
        returned_frame, status = use_generated_dataset_in_fit(frame)
        self.assertEqual(len(returned_frame), 3)
        self.assertIn('Generated dataset is ready for fitting', status)

    def test_request_generated_dataset_transfer_requires_confirmation_when_fit_has_data(self) -> None:
        frame = pd.DataFrame(
            {
                'day_number': [0.0, 1.0],
                'installs': [100.0, 120.0],
                'retention': [0.4, 0.35],
                'retention_mean': [0.41, 0.36],
            }
        )
        preview, status, _warning, _confirm, fit_has_data = request_generated_dataset_transfer(frame, None, True)
        self.assertNotIsInstance(preview, pd.DataFrame)
        self.assertIn('overwrite', status)
        self.assertTrue(fit_has_data)

    def test_confirm_generated_dataset_transfer_overwrites_preview(self) -> None:
        frame = pd.DataFrame(
            {
                'day_number': [0.0, 1.0],
                'installs': [100.0, 120.0],
                'retention': [0.4, 0.35],
                'retention_mean': [0.41, 0.36],
            }
        )
        preview, status, _warning, _confirm, fit_has_data = confirm_generated_dataset_transfer(frame)
        self.assertEqual(len(preview), 2)
        self.assertIn('replaced', status)
        self.assertTrue(fit_has_data)

    def test_fit_uploaded_dataset_accepts_generated_frame(self) -> None:
        frame = pd.DataFrame(
            {
                'day_number': [0.0, 1.0, 2.0, 3.0],
                'installs': [1000.0, 1000.0, 1000.0, 1000.0],
                'retention': [0.45, 0.4, 0.35, 0.3],
                'retention_mean': [0.44, 0.39, 0.34, 0.29],
            }
        )
        figure, predictions, summary, output_path, fit_has_data = fit_uploaded_dataset(
            csv_file=None,
            generated_frame=frame,
            first_day_of_week=0,
            main_function_type='w0',
            chain_function_type='w0',
            connector_type='mul',
            patches_dates='',
            bad_dates='',
            week_function_weights='1, 1, 1, 1, 1, 1, 1',
            training_mode='Fast',
            exclude_patch_dates=True,
        )
        self.assertIsNotNone(figure)
        self.assertEqual(len(predictions), 4)
        self.assertIn('Source: generated dataset', summary)
        self.assertTrue(output_path.endswith('.csv'))
        self.assertTrue(fit_has_data)


if __name__ == '__main__':
    unittest.main()
