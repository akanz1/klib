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
                                       [True, False, 7, pd.NaT]],
                                      columns=['Col1', 'Col2', 'Col3', 'Col4'])

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
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows'][0], 1)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows'][1], 2)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows'][2], 1)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows'][3], 1)

    def test_mv_cols(self):
        # Test missing values for each column
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols'][0], 1)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols'][1], 1)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols'][2], 1)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols'][3], 2)

    def test_mv_rows_ratio(self):
        # Test missing values ratio for each row
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows_ratio'][0], 0.25)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows_ratio'][1], 0.5)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows_ratio'][2], 0.25)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_rows_ratio'][3], 0.25)

        # Test if missing value ratio is between 0 and 1
        self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_rows_ratio'][0] <= 1)
        self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_rows_ratio'][1] <= 1)
        self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_rows_ratio'][2] <= 1)
        self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_rows_ratio'][3] <= 1)

    def test_mv_cols_ratio(self):
        # Test missing values ratio for each row
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols_ratio'][0], 0.25)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols_ratio'][1], 0.25)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols_ratio'][2], 0.25)
        self.assertAlmostEqual(_missing_vals(self.data_mv_df)['mv_cols_ratio'][3], 0.5)

        # Test if missing value ratio is between 0 and 1
        self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_cols_ratio'][0] <= 1)
        self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_cols_ratio'][1] <= 1)
        self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_cols_ratio'][2] <= 1)
        self.assertTrue(0 <= _missing_vals(self.data_mv_df)['mv_cols_ratio'][3] <= 1)


class Test_corr_mat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_corr = pd.DataFrame([[1, 0, 3j, 4],
                                      [3, 4, 5, 6],
                                      ['a', 'b', pd.NA, 'd'],
                                      [5, False, np.nan, pd.NaT]],
                                     columns=['Col1', 'Col2', 'Col3', 'Col4'])

        cls.data_corr_list = [1, 2, -3, 4j, 5, 0]

    def test_output_type(self):
        # Test conversion from pd.io.formats.style.Styler to pd.core.frame.DataFrame
        self.assertTrue(type(corr_mat(self.data_corr)), type(pd.DataFrame))
        self.assertTrue(type(corr_mat(self.data_corr_list)), type(pd.DataFrame))

    def test_output_shape(self):
        # Test for output of equal dimensions
        self.assertEqual(corr_mat(self.data_corr).data.shape[0], corr_mat(self.data_corr).data.shape[1])
        self.assertEqual(corr_mat(self.data_corr_list).data.shape[0], corr_mat(self.data_corr_list).data.shape[1])
