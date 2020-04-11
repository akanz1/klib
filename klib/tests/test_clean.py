import numpy as np
import pandas as pd
import unittest
from klib.clean import drop_missing

if __name__ == '__main__':
    unittest.main()


class Test__missing_vals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_drop_df = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan],
                                         [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                                         [pd.NA, 'b', 'c', 'd', 'e'],
                                         [pd.NA, 6, 7, 8, 9],
                                         [pd.NA, 2, 3, 4, pd.NA],
                                         [pd.NA, 6, 7, pd.NA, pd.NA]])

    def test_drop_missing(self):
        self.assertEqual(drop_missing(self.data_drop_df).shape, (4, 4))

        # Drop further columns
        self.assertEqual(drop_missing(self.data_drop_df, drop_threshold_cols=0.5).shape, (4, 4))
        self.assertEqual(drop_missing(self.data_drop_df, drop_threshold_cols=0.49).shape, (4, 3))
        self.assertEqual(drop_missing(self.data_drop_df, drop_threshold_cols=0).shape, (4, 2))

        # Drop further rows
        self.assertEqual(drop_missing(self.data_drop_df, drop_threshold_rows=0.5).shape, (4, 4))
        self.assertEqual(drop_missing(self.data_drop_df, drop_threshold_rows=0.49).shape, (3, 4))
        self.assertEqual(drop_missing(self.data_drop_df, drop_threshold_rows=0).shape, (2, 4))

    def test__validate_input_0_1(self):
        with self.assertRaises(ValueError):
            drop_missing(self.data_drop_df, drop_threshold_rows=-0.1)

        with self.assertRaises(ValueError):
            drop_missing(self.data_drop_df, drop_threshold_rows=1.1)
