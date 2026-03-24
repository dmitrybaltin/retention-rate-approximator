from __future__ import annotations

import unittest

import torch

from retention_rate_approximator.modeling import ComplexApproximator
from retention_rate_approximator.synthetic import generate_retention_dataset
from retention_rate_approximator.training import custom_mse_loss
from tests.notebook_compat import load_notebook_namespace


class NotebookCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = load_notebook_namespace()

    def test_complex_approximator_forward_matches_notebook(self) -> None:
        notebook_model = self.notebook['ComplexApproximator'](
            first_day_of_week=2,
            patches_dates=[30, 60, 90],
            main_function_type='4',
            chain_functions_type='0',
            connector_type='mul',
            main_function_initial_weights=[0.5, 0.4, 0.05],
            chain_functions_initial_weights=[0.01, 0.02, 0.03],
            week_function_initial_weights=[1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 0.9],
        )
        package_model = ComplexApproximator(
            first_day_of_week=2,
            patches_dates=[30, 60, 90],
            main_function_type='4',
            chain_functions_type='0',
            connector_type='mul',
            main_function_initial_weights=[0.5, 0.4, 0.05],
            chain_functions_initial_weights=[0.01, 0.02, 0.03],
            week_function_initial_weights=[1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 0.9],
        )
        x = torch.arange(0, 120, dtype=torch.float32)

        notebook_predicted = notebook_model(x)
        package_predicted = package_model(x)

        self.assertTrue(torch.allclose(package_predicted, notebook_predicted))

    def test_custom_loss_matches_notebook(self) -> None:
        y_pred = torch.tensor([0.42, 0.37, 0.33], dtype=torch.float32)
        y_true = torch.tensor([0.40, 0.35, 0.31], dtype=torch.float32)
        sample_size = torch.tensor([1000.0, 1200.0, 800.0], dtype=torch.float32)
        notebook_regularizer = lambda: torch.tensor(0.25, dtype=torch.float32)
        package_regularizer = lambda: torch.tensor(0.25, dtype=torch.float32)

        notebook_loss = self.notebook['custom_mse_loss'](
            y_pred,
            y_true,
            sample_size,
            notebook_regularizer,
            100.0,
        )
        package_loss = custom_mse_loss(
            y_pred,
            y_true,
            sample_size,
            package_regularizer,
            100.0,
        )

        self.assertTrue(torch.allclose(package_loss, notebook_loss))

    def test_synthetic_generation_matches_notebook_with_fixed_seed(self) -> None:
        torch.manual_seed(1234)
        self.notebook['np'].random.seed(1234)
        notebook_generated = self.notebook['generate_retention_dataset_2'](
            20,
            2,
            [5, 10, 15],
            '4',
            '0',
            [0.5, 0.4, 0.05],
            [0.01, 0.02, 0.03],
            [1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 0.9],
            1000,
            200,
        )

        torch.manual_seed(1234)
        self.notebook['np'].random.seed(1234)
        package_generated = generate_retention_dataset(
            total_days=20,
            first_day_of_week=2,
            patches_dates=[5, 10, 15],
            main_function_type='4',
            chains_functions_type='0',
            main_function_weights=[0.5, 0.4, 0.05],
            chains_functions_weights=[0.01, 0.02, 0.03],
            week_function_weights=[1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 0.9],
            daily_installs_mean=1000,
            daily_installs_sigma=200,
        )

        notebook_day_numbers = notebook_generated[0]
        notebook_installs = notebook_generated[1]
        notebook_retention = notebook_generated[2]
        notebook_retention_trend = notebook_generated[3]
        notebook_bad_days = notebook_generated[6]

        self.assertTrue(torch.equal(package_generated.day_numbers, notebook_day_numbers))
        self.assertTrue(torch.allclose(package_generated.installs, notebook_installs))
        self.assertTrue(torch.allclose(package_generated.retention, notebook_retention))
        self.assertTrue(torch.allclose(package_generated.retention_trend, notebook_retention_trend))
        self.assertEqual(package_generated.bad_days, notebook_bad_days)


if __name__ == '__main__':
    unittest.main()