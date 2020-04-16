import numpy as np
import pandas as pd
import unittest
from klib.clean import drop_missing

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
