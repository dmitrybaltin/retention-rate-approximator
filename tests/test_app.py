from __future__ import annotations

import unittest

from app import _build_dataset_download_path


class AppTests(unittest.TestCase):
    def test_download_path_preserves_csv_extension(self) -> None:
        path = _build_dataset_download_path('demo_dataset.csv')
        self.assertEqual(path.name, 'demo_dataset.csv')
        self.assertEqual(path.suffix, '.csv')


if __name__ == '__main__':
    unittest.main()