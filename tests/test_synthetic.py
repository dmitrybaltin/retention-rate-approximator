from __future__ import annotations

import unittest

from retention_rate_approximator.synthetic import generate_retention_dataset


class SyntheticTests(unittest.TestCase):
    def test_generate_retention_dataset_shapes(self) -> None:
        generated = generate_retention_dataset(
            total_days=20,
            first_day_of_week=0,
            patches_dates=[5, 10],
            main_function_type="4",
            chains_functions_type="0",
            main_function_weights=[0.5, 0.4, 0.05],
            chains_functions_weights=[0.01, 0.02],
            week_function_weights=[1.0, 1.0, 1.0, 1.0, 1.05, 1.05, 0.9],
            daily_installs_mean=1000,
            daily_installs_sigma=200,
        )

        self.assertEqual(tuple(generated.day_numbers.shape), (20,))
        self.assertEqual(tuple(generated.installs.shape), (20,))
        self.assertEqual(tuple(generated.retention.shape), (20,))
        self.assertEqual(len(generated.modeled_chains), 2)
        self.assertEqual(generated.patches_dates, [5, 10])


if __name__ == "__main__":
    unittest.main()
