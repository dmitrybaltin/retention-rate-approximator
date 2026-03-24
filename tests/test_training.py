from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd
import torch

from retention_rate_approximator.data import RetentionDataset
from retention_rate_approximator.training import TrainingPhase, train_retention_model


class TrainingTests(unittest.TestCase):
    def test_train_retention_model_returns_predictions(self) -> None:
        dataset = RetentionDataset(
            day_numbers=torch.arange(0, 12, dtype=torch.float32),
            installs=torch.full((12,), 1000.0),
            retention=torch.linspace(0.45, 0.20, 12),
            retention_mean=torch.linspace(0.44, 0.21, 12),
            first_date=datetime(2025, 1, 1),
            frame=pd.DataFrame(),
        )

        result = train_retention_model(
            dataset=dataset,
            first_day_of_week=0,
            main_function_type="w0",
            chain_function_type="w0",
            connector_type="mul",
            patches_dates=[4, 8],
            week_function_initial_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            training_strategy=[
                TrainingPhase("Adam", 3, True, False, 0.01),
                TrainingPhase("LBFGS", 1, True, False, 0.01),
            ],
        )

        self.assertEqual(tuple(result.predicted.shape), (12,))
        self.assertEqual(tuple(result.predicted_trend.shape), (12,))
        self.assertEqual(tuple(result.predicted_week.shape), (12,))
        self.assertGreaterEqual(len(result.loss_history), 4)

    def test_train_retention_model_rejects_empty_training_rows(self) -> None:
        dataset = RetentionDataset(
            day_numbers=torch.tensor([0.0, 1.0]),
            installs=torch.tensor([100.0, 100.0]),
            retention=torch.tensor([0.4, 0.3]),
            retention_mean=torch.tensor([0.4, 0.3]),
            first_date=datetime(2025, 1, 1),
            frame=pd.DataFrame(),
        )

        with self.assertRaises(ValueError):
            train_retention_model(
                dataset=dataset,
                first_day_of_week=0,
                main_function_type="w0",
                chain_function_type="w0",
                connector_type="mul",
                patches_dates=[],
                bad_dates=[0, 1],
            )


if __name__ == "__main__":
    unittest.main()
