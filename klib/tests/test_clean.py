import numpy as np
import pandas as pd
import unittest
from klib.clean import drop_missing
from klib.clean import convert_datatypes

if __name__ == '__main__':
    unittest.main()


class Test_drop_missing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_data_drop = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                         [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                                         [pd.NA, 'b', 'c', 'd', 'e'],
                                         [pd.NA, 6, 7, 8, 9],
                                         [pd.NA, 2, 3, 4, pd.NA],
                                         [pd.NA, 6, 7, pd.NA, pd.NA]])

    def test_drop_missing(self):
        self.assertEqual(drop_missing(self.df_data_drop).shape, (4, 4))

        # Drop further columns based on threshold
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_cols=0.5).shape, (4, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_cols=0.49).shape, (4, 3))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_cols=0).shape, (4, 2))

        # Drop further rows based on threshold
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.5).shape, (4, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0.49).shape, (3, 4))
        self.assertEqual(drop_missing(self.df_data_drop, drop_threshold_rows=0).shape, (2, 4))


class Test_convert_dtypes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df_data_convert = pd.DataFrame([[1, 7, 'y', 'x', pd.NA, 'v'],
                                            [3, 8, 'd', 'e', pd.NA, 'v'],
                                            [5, 7, 'o', 'z', pd.NA, 'v'],
                                            [1, 7, 'u', 'f', pd.NA, 'p'],
                                            [1, 7, 'u', 'f', pd.NA, 'p'],
                                            [2, 7, 'g', 'a', pd.NA, 'p']])

    def test_convert_dtypes(self):
        expected_results = ['Int64', 'Int64', 'string', 'string', 'category', 'category']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert, cat_threshold=0.4).dtypes[i], expected_results[i])

        expected_results = ['Int64', 'Int64', 'string', 'string', 'object', 'string']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert).dtypes[i], expected_results[i])

        expected_results = ['Int64', 'Int64', 'string', 'string', 'object', 'category']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert, cat_threshold=0.5,
                                               cat_exclude=[4]).dtypes[i], expected_results[i])

        expected_results = ['Int64', 'Int64', 'string', 'category', 'object', 'category']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert, cat_threshold=0.95,
                                               cat_exclude=[2, 4]).dtypes[i], expected_results[i])

        expected_results = ['Int64', 'Int64', 'string', 'string', 'object', 'string']
        for i, _ in enumerate(expected_results):
            self.assertEqual(convert_datatypes(self.df_data_convert, category=False,
                                               cat_threshold=0.95, cat_exclude=[2, 4]).dtypes[i], expected_results[i])
