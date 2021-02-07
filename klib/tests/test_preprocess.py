import pandas as pd
import unittest

from ..preprocess import train_dev_test_split


class Test_train_dev_test_split(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_split = pd.DataFrame(
            [
                [1, 2, 3, 4, "a"],
                [2, 4, 5, 6, "b"],
                [3, 4, 2, 1, "c"],
                [4, 0, 3, 4, "a"],
                [5, 4, 5, 6, "b"],
                [6, 4, 2, 1, "c"],
                [7, 0, 3, 4, "a"],
                [8, 4, 5, 6, "b"],
                [9, 4, 2, 1, "c"],
                [10, 2, 1, 5, "b"],
            ],
            columns=["Col1", "Col2", "Col3", "Col4", "Col5"],
        )
        cls.data_target = pd.Series([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])

    def test_train_dev_test_split_col(self):
        # Test the propper splitting in train, dev and test sets

        expected_results = [(8, 4), (1, 4), (1, 4), (8,), (1,), (1,)]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                train_dev_test_split(self.data_split, "Col2", random_state=1234)[
                    i
                ].shape,
                expected_results[i],
            )

        expected_results = [(8, 4), (2, 4), (8,), (2,)]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                train_dev_test_split(
                    self.data_split, target="Col2", dev_size=0, test_size=0.2
                )[i].shape,
                expected_results[i],
            )

        expected_results = [(5, 4), (5, 4), (5,), (5,)]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                train_dev_test_split(
                    self.data_split, target="Col2", dev_size=0.5, test_size=0
                )[i].shape,
                expected_results[i],
            )

    def test_train_dev_test_split_series(self):
        # Test the propper splitting in train, dev and test sets

        expected_results = [(6, 5), (2, 5), (2, 5), (6,), (2,), (2,)]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                train_dev_test_split(
                    self.data_split,
                    target=self.data_target,
                    dev_size=0.2,
                    test_size=0.2,
                )[i].shape,
                expected_results[i],
            )

        expected_results = [(8, 5), (2, 5), (8,), (2,)]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                train_dev_test_split(
                    self.data_split, target=self.data_target, dev_size=0, test_size=0.2
                )[i].shape,
                expected_results[i],
            )

        expected_results = [(5, 5), (5, 5), (5,), (5,)]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                train_dev_test_split(
                    self.data_split, target=self.data_target, dev_size=0.5, test_size=0
                )[i].shape,
                expected_results[i],
            )
