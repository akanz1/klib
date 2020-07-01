import numpy as np
import pandas as pd
import unittest
from ..clean import (drop_missing,
                     convert_datatypes,
                     pool_duplicate_subsets)


class Test_drop_missing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_data_drop = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                         [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                                         [pd.NA, 'b', 'c', 'd', 'e'],
                                         [pd.NA, 6, 7, 8, 9],
                                         [pd.NA, 2, 3, 4, pd.NA],
                                         [pd.NA, 6, 7, pd.NA, pd.NA]], columns=['c1', 'c2', 'c3', 'c4', 'c5'])

    def test_drop_missing(self):
        self.assertEqual(drop_missing(self.df_data_drop).shape, (4, 4))

        # Drop further columns based on threshold
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_cols=0.5).shape, (4, 3))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_cols=0.5, col_exclude=['c1']).shape, (4, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_cols=0.49).shape, (4, 2))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_cols=0).shape, (0, 0))

        # Drop further rows based on threshold
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.67).shape, (4, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.5).shape, (4, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.49).shape, (3, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.25).shape, (3, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.24).shape, (2, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.24, col_exclude=['c1']).shape, (2, 5))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.24, col_exclude=['c2']).shape, (2, 4))


class Test_convert_dtypes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_data_convert = pd.DataFrame([[1, 7.0, 'y', 'x', pd.NA, 'v'],
                                            [3, 8.0, 'd', 'e', pd.NA, 'v'],
                                            [5, 7.0, 'o', 'z', pd.NA, 'v'],
                                            [1, 7.0, 'u', 'f', pd.NA, 'p'],
                                            [1, 7.0, 'u', 'f', pd.NA, 'p'],
                                            [2, 7.0, 'g', 'a', pd.NA, 'p']])

    def test_convert_dtypes(self):
        expected_results = ['Int8', 'Float32', 'string', 'string', 'category', 'category']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert, cat_threshold=0.4).dtypes[i], expected_results[i])

        expected_results = ['Int8', 'Float32', 'string', 'string', 'object', 'string']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert).dtypes[i], expected_results[i])

        expected_results = ['Int8', 'Float32', 'string', 'string', 'object', 'category']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert, cat_threshold=0.5,
                                               cat_exclude=[4]).dtypes[i], expected_results[i])

        expected_results = ['Int8', 'Float32', 'string', 'category', 'object', 'category']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert, cat_threshold=0.95,
                                               cat_exclude=[2, 4]).dtypes[i], expected_results[i])

        expected_results = ['Int8', 'Float32', 'string', 'string', 'object', 'string']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert, category=False,
                                               cat_threshold=0.95, cat_exclude=[2, 4]).dtypes[i], expected_results[i])


class Test_pool_duplicate_subsets(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_data_subsets = pd.DataFrame([[1, 7, 'd', 'x', pd.NA, 'v'],
                                            [1, 8, 'd', 'e', pd.NA, 'v'],
                                            [2, 7, 'g', 'z', pd.NA, 'v'],
                                            [1, 7, 'u', 'f', pd.NA, 'p'],
                                            [1, 7, 'u', 'z', pd.NA, 'p'],
                                            [2, 7, 'g', 'z', pd.NA, 'p']])

    def test_pool_duplicate_subsets(self):
        self.assertEqual(pool_duplicate_subsets(self.df_data_subsets).shape, (6, 3))
        self.assertEqual(pool_duplicate_subsets(self.df_data_subsets, col_dupl_thresh=1).shape, (6, 6))
        self.assertEqual(pool_duplicate_subsets(self.df_data_subsets, subset_thresh=0).shape, (6, 2))
