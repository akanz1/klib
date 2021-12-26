import unittest

import numpy as np
import pandas as pd

from klib.clean import (
    clean_column_names,
    convert_datatypes,
    data_cleaning,
    drop_missing,
    pool_duplicate_subsets,
)


class Test_clean_column_names(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df1 = pd.DataFrame(
            {
                "Asd 5$ & (3€)": [1, 2, 3],
                "3+3": [2, 3, 4],
                "AsdFer #9": [3, 4, 5],
                '"asdäöüß"': [5, 6, 7],
                "dupli": [5, 6, 8],
                "also": [9, 2, 7],
                "-Ä-__________!?:;some/(... \n ..))(++$%/name/    -.....": [2, 3, 7],
            }
        )
        cls.df2 = pd.DataFrame(
            {
                "dupli": [3, 2, 1],
                "also": [4, 5, 7],
                "verylongColumnNamesareHardtoRead": [9, 2, 7],
                "< #total@": [2, 6, 4],
                "count >= 10": [6, 3, 2],
            }
        )
        cls.df_clean_column_names = pd.concat([cls.df1, cls.df2], axis=1)

    def test_clean_column_names(self):
        expected_results = [
            "asd_5_dollar_and_3_euro",
            "3_plus_3",
            "asd_fer_hash_9",
            "asdaeoeuess",
            "dupli",
            "also",
            "ae_some_plus_plus_dollar_percent_name",
            "dupli_7",
            "also_8",
            "verylong_column_namesare_hardto_read",
            "smaller_hash_total_at",
            "count_larger_equal_10",
        ]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                clean_column_names(self.df_clean_column_names).columns[i],
                expected_results[i],
            )
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                clean_column_names(self.df_clean_column_names, hints=False).columns[i],
                expected_results[i],
            )


class Test_drop_missing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_data_drop = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                [pd.NA, "b", "c", "d", "e"],
                [pd.NA, 6, 7, 8, 9],
                [pd.NA, 2, 3, 4, pd.NA],
                [pd.NA, 6, 7, pd.NA, pd.NA],
            ],
            columns=["c1", "c2", "c3", "c4", "c5"],
        )

    def test_drop_missing(self):
        self.assertEqual(drop_missing(self.df_data_drop).shape, (4, 4))

        # Drop further columns based on threshold
        self.assertEqual(
            drop_missing(self.df_data_drop, drop_threshold_cols=0.5).shape, (4, 3)
        )
        self.assertEqual(
            drop_missing(
                self.df_data_drop, drop_threshold_cols=0.5, col_exclude=["c1"]
            ).shape,
            (4, 4),
        )
        self.assertEqual(
            drop_missing(self.df_data_drop, drop_threshold_cols=0.49).shape, (4, 2)
        )
        self.assertEqual(
            drop_missing(self.df_data_drop, drop_threshold_cols=0).shape, (0, 0)
        )

        # Drop further rows based on threshold
        self.assertEqual(
            drop_missing(self.df_data_drop, drop_threshold_rows=0.67).shape, (4, 4)
        )
        self.assertEqual(
            drop_missing(self.df_data_drop, drop_threshold_rows=0.5).shape, (4, 4)
        )
        self.assertEqual(
            drop_missing(self.df_data_drop, drop_threshold_rows=0.49).shape, (3, 4)
        )
        self.assertEqual(
            drop_missing(self.df_data_drop, drop_threshold_rows=0.25).shape, (3, 4)
        )
        self.assertEqual(
            drop_missing(self.df_data_drop, drop_threshold_rows=0.24).shape, (2, 4)
        )
        self.assertEqual(
            drop_missing(
                self.df_data_drop, drop_threshold_rows=0.24, col_exclude=["c1"]
            ).shape,
            (2, 5),
        )
        self.assertEqual(
            drop_missing(
                self.df_data_drop, drop_threshold_rows=0.24, col_exclude=["c2"]
            ).shape,
            (2, 4),
        )
        self.assertEqual(
            drop_missing(
                self.df_data_drop, drop_threshold_rows=0.51, col_exclude=["c1"]
            ).shape,
            (3, 5),
        )


class Test_data_cleaning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_data_cleaning = pd.DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                [pd.NA, "b", 6, "d", "e"],
                [pd.NA, "b", 7, 8, 9],
                [pd.NA, "c", 3, 4, pd.NA],
                [pd.NA, "d", 7, pd.NA, pd.NA],
            ],
            columns=["c1", "c2", "c3", "c4", "c5"],
        )

    def test_data_cleaning(self):
        self.assertEqual(data_cleaning(self.df_data_cleaning, show="all").shape, (4, 4))
        # c1 will be dropped despite in col_exclude because it is single valued
        self.assertEqual(
            data_cleaning(self.df_data_cleaning, col_exclude=["c1"]).shape, (4, 4)
        )

        expected_results = ["string", "int8", "O", "O"]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                data_cleaning(self.df_data_cleaning, convert_dtypes=True).dtypes[i],
                expected_results[i],
            )


class Test_convert_dtypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_data_convert = pd.DataFrame(
            [
                [1, 7.0, "y", "x", pd.NA, "v"],
                [3, 8.0, "d", "e", pd.NA, "v"],
                [5, 7.0, "o", "z", pd.NA, "v"],
                [1, 7.0, "u", "f", pd.NA, "p"],
                [1, 7.0, "u", "f", pd.NA, "p"],
                [2, 7.0, "g", "a", pd.NA, "p"],
            ]
        )

    def test_convert_dtypes(self):
        expected_results = [
            "int8",
            "Float32",
            "string",
            "string",
            "category",
            "category",
        ]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                convert_datatypes(self.df_data_convert, cat_threshold=0.4).dtypes[i],
                expected_results[i],
            )

        expected_results = [
            "int8",
            "Float32",
            "string",
            "string",
            "object",
            "string",
        ]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                convert_datatypes(self.df_data_convert).dtypes[i], expected_results[i]
            )

        expected_results = [
            "int8",
            "Float32",
            "string",
            "string",
            "object",
            "category",
        ]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                convert_datatypes(
                    self.df_data_convert, cat_threshold=0.5, cat_exclude=[4]
                ).dtypes[i],
                expected_results[i],
            )

        expected_results = [
            "int8",
            "Float32",
            "string",
            "category",
            "object",
            "category",
        ]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                convert_datatypes(
                    self.df_data_convert, cat_threshold=0.95, cat_exclude=[2, 4]
                ).dtypes[i],
                expected_results[i],
            )

        expected_results = ["int8", "Float32", "string", "string", "object", "string"]
        for i, _ in enumerate(expected_results):
            self.assertEqual(
                convert_datatypes(
                    self.df_data_convert,
                    category=False,
                    cat_threshold=0.95,
                    cat_exclude=[2, 4],
                ).dtypes[i],
                expected_results[i],
            )


class Test_pool_duplicate_subsets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_data_subsets = pd.DataFrame(
            [
                [1, 7, "d", "x", pd.NA, "v"],
                [1, 8, "d", "e", pd.NA, "v"],
                [2, 7, "g", "z", pd.NA, "v"],
                [1, 7, "u", "f", pd.NA, "p"],
                [1, 7, "u", "z", pd.NA, "p"],
                [2, 7, "g", "z", pd.NA, "p"],
            ],
            columns=["c1", "c2", "c3", "c4", "c5", "c6"],
        )

    def test_pool_duplicate_subsets(self):
        self.assertEqual(pool_duplicate_subsets(self.df_data_subsets).shape, (6, 3))
        self.assertEqual(
            pool_duplicate_subsets(self.df_data_subsets, col_dupl_thresh=1).shape,
            (6, 6),
        )

        self.assertEqual(
            pool_duplicate_subsets(self.df_data_subsets, subset_thresh=0).shape, (6, 2)
        )

        self.assertEqual(
            pool_duplicate_subsets(self.df_data_subsets, return_details=True)[0].shape,
            (6, 3),
        )
        self.assertEqual(
            pool_duplicate_subsets(self.df_data_subsets, return_details=True)[1],
            ["c1", "c2", "c3", "c5"],
        )

        self.assertEqual(
            pool_duplicate_subsets(self.df_data_subsets, exclude=["c1"]).shape,
            (6, 4),
        )

        self.assertEqual(
            pool_duplicate_subsets(
                self.df_data_subsets, exclude=["c1"], return_details=True
            )[1],
            ["c2", "c5", "c6"],
        )
