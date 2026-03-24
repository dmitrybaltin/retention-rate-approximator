from __future__ import annotations

import unittest

import torch

from retention_rate_approximator.modeling import ApproximatorsFactory, ComplexApproximator


class ModelingTests(unittest.TestCase):
    def test_factory_returns_expected_defaults(self) -> None:
        weights = ApproximatorsFactory.get_main_function_weights_number("w0-w1*x/(w2+x)")
        self.assertEqual(weights, (0.5, 0.25, 10.0))

    def test_complex_approximator_forward_shapes(self) -> None:
        model = ComplexApproximator(
            first_day_of_week=0,
            patches_dates=[5, 10],
            main_function_type="w0",
            chain_functions_type="w0",
            connector_type="mul",
            week_function_initial_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        x = torch.arange(0, 14, dtype=torch.float32)
        predicted = model(x)
        trend = model.forward_trend_function(x)
        week = model.forward_week_function(x)

        self.assertEqual(predicted.shape, x.shape)
        self.assertEqual(trend.shape, x.shape)
        self.assertEqual(week.shape, x.shape)
        self.assertTrue(torch.allclose(predicted, trend * week))

    def test_patch_dates_are_normalized(self) -> None:
        model = ComplexApproximator(
            first_day_of_week=0,
            patches_dates=[0, 10, 10, 3, -1],
            main_function_type="w0",
            chain_functions_type="w0",
        )
        self.assertEqual(model.patches_dates, [3, 10])


if __name__ == "__main__":
    unittest.main()
