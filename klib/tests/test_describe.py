import numpy as np
import pandas as pd
import unittest
from klib.describe import _missing_vals, corr_mat

if __name__ == '__main__':
    unittest.main()


class Test__missing_vals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_mv_df = pd.DataFrame([[1, np.nan, 3, 4],
                                       [None, 4, 5, None],
                                       ['a', 'b', pd.NA, 'd'],
                                       [True, False, 7, pd.NaT]])

        cls.data_mv_array = np.array([[1, np.nan, 3, 4],
                                      [None, 4, 5, None],
                                      ['a', 'b', pd.NA, 'd'],
                                      [True, False, 7, pd.NaT]])

        cls.data_mv_list = [[1, np.nan, 3, 4],
                            [None, 4, 5, None],
                            ['a', 'b', pd.NA, 'd'],
                            [True, False, 7, pd.NaT]]

    def test_mv_total(self):
        # Test total missing values
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_total'], 5)
        self.assertAlmostEqual(_missing_vals(self.data_mv_array)['mv_total'], 5)
        self.assertAlmostEqual(_missing_vals(self.data_mv_list)['mv_total'], 5)

    def test_mv_rows(self):
        # Test missing values for each row
        expected_results = [1, 2, 1, 1]
        for i, _ in enumerate(expected_results):
            self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows'][i], expected_results[i])

    def test_mv_cols(self):
        # Test missing values for each column
        expected_results = [1, 1, 1, 2]
        for i, _ in enumerate(expected_results):
            self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols'][i], expected_results[i])

    def test_mv_rows_ratio(self):
        # Test missing values ratio for each row
        expected_results = [0.25, 0.5, 0.25, 0.25]
        for i, _ in enumerate(expected_results):
            self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows_ratio'][i], expected_results[i])

        # Test if missing value ratio is between 0 and 1
        for i in range(len(self.data_mv_df)):
            self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_rows_ratio'][i] <= 1)

    def test_mv_cols_ratio(self):
        # Test missing values ratio for each column
        expected_results = [1/4, 0.25, 0.25, 0.5]
        for i, _ in enumerate(expected_results):
            self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols_ratio'][i], expected_results[i])

        # Test if missing value ratio is between 0 and 1
        for i in range(len(self.data_mv_df)):
            self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_cols_ratio'][i] <= 1)


class Test_corr_mat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_corr = pd.DataFrame([[1, 0, 3, 4],
                                      [3, 4, 5, 6],
                                      ['a', 'b', pd.NA, 'd'],
                                      [5, False, np.nan, pd.NaT]],
                                     columns=['Col1', 'Col2', 'Col3', 'Col4'])

        cls.data_corr_list = [1, 2, -3, 4, 5, 0]

    def test_output_type(self):
        # Test conversion from pd.io.formats.style.Styler to pd.core.frame.DataFrame
        self.assertTrue(type(corr_mat(self.data_corr)), type(pd.DataFrame))
        self.assertTrue(type(corr_mat(self.data_corr_list)), type(pd.DataFrame))

    def test_output_shape(self):
        # Test for output of equal dimensions
        self.assertEqual(corr_mat(self.data_corr).data.shape[0], corr_mat(self.data_corr).data.shape[1])
        self.assertEqual(corr_mat(self.data_corr_list).data.shape[0], corr_mat(self.data_corr_list).data.shape[1])
