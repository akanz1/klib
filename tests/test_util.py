import unittest

import numpy as np
import pandas as pd
from klib.utils import _corr_selector
from klib.utils import _drop_duplicates
from klib.utils import _missing_vals
from klib.utils import _validate_input_bool
from klib.utils import _validate_input_int
from klib.utils import _validate_input_num_data
from klib.utils import _validate_input_range
from klib.utils import _validate_input_smaller
from klib.utils import _validate_input_sum_larger
from klib.utils import _validate_input_sum_smaller


class Test__corr_selector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_data_corr = pd.DataFrame(
            [
                [1, 7, 2, 2, 4, 7],
                [3, 8, 3, 3, 7, 1],
                [5, 7, 9, 5, 1, 4],
                [1, 7, 8, 6, 1, 8],
                [1, 7, 5, 6, 2, 6],
                [2, 7, 3, 3, 5, 3],
            ],
        )

        cls.target = pd.Series([1, 2, 4, 7, 4, 2])

    def test__corr_selector_matrix(self):
        assert _corr_selector(self.df_data_corr.corr()).shape == (6, 6)
        assert (
            _corr_selector(self.df_data_corr.corr(), split="pos").isna().sum().sum()
            == 18
        )
        assert (
            _corr_selector(self.df_data_corr.corr(), split="pos", threshold=0.5)
            .isna()
            .sum()
            .sum()
            == 26
        )
        assert (
            _corr_selector(self.df_data_corr.corr(), split="neg", threshold=-0.75)
            .isna()
            .sum()
            .sum()
            == 32
        )
        assert (
            _corr_selector(self.df_data_corr.corr(), split="high", threshold=0.15)
            .isna()
            .sum()
            .sum()
            == 4
        )
        assert (
            _corr_selector(self.df_data_corr.corr(), split="low", threshold=0.85)
            .isna()
            .sum()
            .sum()
            == 6
        )

    def test__corr_selector_label(self):
        assert _corr_selector(self.df_data_corr.corrwith(self.target)).shape == (6,)
        assert (
            _corr_selector(self.df_data_corr.corrwith(self.target), split="pos")
            .isna()
            .sum()
            == 3
        )
        assert (
            _corr_selector(
                self.df_data_corr.corrwith(self.target),
                split="pos",
                threshold=0.8,
            )
            .isna()
            .sum()
            == 4
        )
        assert (
            _corr_selector(
                self.df_data_corr.corrwith(self.target),
                split="neg",
                threshold=-0.7,
            )
            .isna()
            .sum()
            == 5
        )
        assert (
            _corr_selector(
                self.df_data_corr.corrwith(self.target),
                split="high",
                threshold=0.2,
            )
            .isna()
            .sum()
            == 1
        )
        assert (
            _corr_selector(
                self.df_data_corr.corrwith(self.target),
                split="low",
                threshold=0.8,
            )
            .isna()
            .sum()
            == 2
        )


class Test__drop_duplicates(unittest.TestCase):
    @classmethod
    def setUpClass(cls: pd.DataFrame) -> pd.DataFrame:
        cls.data_dupl_df = pd.DataFrame(
            [
                [pd.NA, pd.NA, pd.NA, pd.NA],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [1, 2, 3, pd.NA],
                [pd.NA, pd.NA, pd.NA, pd.NA],
            ],
        )

    def test__drop_dupl(self):
        # Test dropping of duplicate rows
        self.assertAlmostEqual(_drop_duplicates(self.data_dupl_df)[0].shape, (4, 4))
        # Test if the resulting DataFrame is equal to using the pandas method
        assert _drop_duplicates(self.data_dupl_df)[0].equals(
            self.data_dupl_df.drop_duplicates().reset_index(drop=True),
        )
        # Test number of duplicates
        assert len(_drop_duplicates(self.data_dupl_df)[1]) == 3


class Test__missing_vals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_mv_list = [
            [1, np.nan, 3, 4],
            [None, 4, 5, None],
            ["a", "b", pd.NA, "d"],
            [True, False, 7, pd.NaT],
        ]

        cls.data_mv_df = pd.DataFrame(cls.data_mv_list)

        cls.data_mv_array = np.array(cls.data_mv_list)

    def test_mv_total(self):
        # Test total missing values
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)["mv_total"], 5)
        self.assertAlmostEqual(_missing_vals(self.data_mv_array)["mv_total"], 5)
        self.assertAlmostEqual(_missing_vals(self.data_mv_list)["mv_total"], 5)

    def test_mv_rows(self):
        # Test missing values for each row
        expected_results = [1, 2, 1, 1]
        for i, result in enumerate(expected_results):
            self.assertAlmostEqual(_missing_vals(self.data_mv_df)["mv_rows"][i], result)

    def test_mv_cols(self):
        # Test missing values for each column
        expected_results = [1, 1, 1, 2]
        for i, result in enumerate(expected_results):
            self.assertAlmostEqual(_missing_vals(self.data_mv_df)["mv_cols"][i], result)

    def test_mv_rows_ratio(self):
        # Test missing values ratio for each row
        expected_results = [0.25, 0.5, 0.25, 0.25]
        for i, result in enumerate(expected_results):
            self.assertAlmostEqual(
                _missing_vals(self.data_mv_df)["mv_rows_ratio"][i],
                result,
            )

        # Test if missing value ratio is between 0 and 1
        for i, _ in enumerate(self.data_mv_df):
            assert 0 <= _missing_vals(self.data_mv_df)["mv_rows_ratio"][i] <= 1

    def test_mv_cols_ratio(self):
        # Test missing values ratio for each column
        expected_results = [1 / 4, 0.25, 0.25, 0.5]
        for i, result in enumerate(expected_results):
            self.assertAlmostEqual(
                _missing_vals(self.data_mv_df)["mv_cols_ratio"][i],
                result,
            )

        # Test if missing value ratio is between 0 and 1
        for i, _ in enumerate(self.data_mv_df):
            assert 0 <= _missing_vals(self.data_mv_df)["mv_cols_ratio"][i] <= 1


class Test__validate_input(unittest.TestCase):
    def test__validate_input_bool(self):
        # Raises an exception if the input is not boolean
        with self.assertRaises(TypeError):
            _validate_input_bool("True", None)
        with self.assertRaises(TypeError):
            _validate_input_bool(None, None)
        with self.assertRaises(TypeError):
            _validate_input_bool(1, None)

    def test__validate_input_int(self):
        # Raises an exception if the input is not an integer
        with self.assertRaises(TypeError):
            _validate_input_int(1.1, None)
        with self.assertRaises(TypeError):
            _validate_input_int([1], None)
        with self.assertRaises(TypeError):
            _validate_input_int("1", None)

    def test__validate_input_smaller(self):
        # Raises an exception if the first value is larger than the second
        with self.assertRaises(ValueError):
            _validate_input_smaller(0.3, 0.2, None)
        with self.assertRaises(ValueError):
            _validate_input_smaller(3, 2, None)
        with self.assertRaises(ValueError):
            _validate_input_smaller(5, -3, None)

    def test__validate_input_range(self):
        with self.assertRaises(ValueError):
            _validate_input_range(-0.1, "value -0.1", 0, 1)

        with self.assertRaises(ValueError):
            _validate_input_range(1.1, "value 1.1", 0, 1)

        with self.assertRaises(TypeError):
            _validate_input_range("1", "value string", 0, 1)

    def test__validate_input_sum_smaller(self):
        with self.assertRaises(ValueError):
            _validate_input_sum_smaller(1, "Test Sum <= 1", 1.01)
        with self.assertRaises(ValueError):
            _validate_input_sum_smaller(1, "Test Sum <= 1", 0.3, 0.2, 0.4, 0.5)
        with self.assertRaises(ValueError):
            _validate_input_sum_smaller(-1, "Test Sum <= -1", -0.2, -0.7)
        with self.assertRaises(ValueError):
            _validate_input_sum_smaller(10, "Test Sum <= 10", 20, -11, 2)

    def test__validate_input_sum_larger(self):
        with self.assertRaises(ValueError):
            _validate_input_sum_larger(1, "Test Sum >= 1", 0.99)
        with self.assertRaises(ValueError):
            _validate_input_sum_larger(1, "Test Sum >= 1", 0.9, 0.05)
        with self.assertRaises(ValueError):
            _validate_input_sum_larger(-2, "Test Sum >=-2", -3)
        with self.assertRaises(ValueError):
            _validate_input_sum_larger(7, "Test Sum >= 7", 1, 2, 3)

    def test__validate_input_num_data(self):
        with self.assertRaises(TypeError):
            _validate_input_num_data(pd.DataFrame({"col1": ["a", "b", "c"]}), None)

        _validate_input_num_data(
            pd.DataFrame({"col1": [1, 2, 3]}),
            None,
        )  # No exception
