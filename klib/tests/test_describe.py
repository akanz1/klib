import numpy as np
import pandas as pd
import unittest
from klib.describe import _missing_vals

if __name__ == '__main__':
    unittest.main()

data_mv = pd.DataFrame([[1, np.nan, 3, 4],
                        [None, 4, 5, ],
                        ['a', 'b', pd.NA, 'd'],
                        [True, False, 7, pd.NaT]],
                       columns=['Col1', 'Col2', 'Col3', 'Col4'])


class Test__missing_vals(unittest.TestCase):

    def test_mv_total(self):
        # Test total missing values
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_total'], 5)

    def test_mv_rows(self):
        # Test missing values for each row
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_rows'][0], 1)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_rows'][1], 2)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_rows'][2], 1)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_rows'][2], 1)

    def test_mv_cols(self):
        # Test missing values for each column
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_cols'][0], 1)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_cols'][1], 1)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_cols'][2], 1)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_cols'][3], 2)

    def test_mv_rows_ratio(self):
        # Test missing values ratio for each row
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_rows_ratio'][0], 0.25)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_rows_ratio'][1], 0.5)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_rows_ratio'][2], 0.25)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_rows_ratio'][3], 0.25)

    def test_mv_cols_ratio(self):
        # Test missing values ratio for each row
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_cols_ratio'][0], 0.25)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_cols_ratio'][1], 0.25)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_cols_ratio'][2], 0.25)
        self.assertAlmostEqual(_missing_vals(data_mv)['mv_cols_ratio'][3], 0.5)
