from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

import torch

from retention_rate_approximator.data import load_retention_csv, save_retention_csv


class DataTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        tmp_dir = Path('tests') / '.tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / 'retention.csv'
        try:
            saved_path = save_retention_csv(
                path=csv_path,
                day_numbers=torch.tensor([0.0, 1.0, 2.0]),
                installs=torch.tensor([100.0, 110.0, 120.0]),
                retention=torch.tensor([0.4, 0.35, 0.3]),
                retention_mean=torch.tensor([0.41, 0.36, 0.31]),
                release_date=datetime(2025, 1, 1),
            )

            dataset = load_retention_csv(saved_path)

            self.assertTrue(saved_path.exists())
            self.assertEqual(dataset.day_numbers.tolist(), [0.0, 1.0, 2.0])
            self.assertEqual(dataset.installs.tolist(), [100.0, 110.0, 120.0])
            self.assertEqual(dataset.first_date, datetime(2025, 1, 1))
            self.assertEqual(list(dataset.frame.columns), ['date', 'day_number', 'installs', 'retention', 'retention_mean'])
        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_save_without_release_date_uses_day_number_column(self) -> None:
        tmp_dir = Path('tests') / '.tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / 'demo.csv'
        try:
            saved_path = save_retention_csv(
                path=csv_path,
                day_numbers=torch.tensor([0.0, 1.0, 2.0]),
                installs=torch.tensor([100.0, 110.0, 120.0]),
                retention=torch.tensor([0.4, 0.35, 0.3]),
                retention_mean=torch.tensor([0.41, 0.36, 0.31]),
            )
            dataset = load_retention_csv(saved_path)

            self.assertEqual(dataset.day_numbers.tolist(), [0.0, 1.0, 2.0])
            self.assertEqual(list(dataset.frame.columns), ['day_number', 'installs', 'retention', 'retention_mean'])
        finally:
            if csv_path.exists():
                csv_path.unlink()


if __name__ == '__main__':
    unittest.main()