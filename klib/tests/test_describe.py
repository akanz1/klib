import numpy as np
import pandas as pd
import unittest
from klib.describe import corr_mat

if __name__ == '__main__':
    unittest.main()


class Test_corr_mat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_corr_df = pd.DataFrame([[1, 0, 3, 4],
                                         [3, 4, 5, 6],
                                         ['a', 'b', pd.NA, 'd'],
                                         [5, False, np.nan, pd.NaT]],
                                        columns=['Col1', 'Col2', 'Col3', 'Col4'])

        cls.data_corr_list = [1, 2, -3, 4, 5, 0]

    def test_output_type(self):
        # Test conversion from pd.io.formats.style.Styler to pd.core.frame.DataFrame
        self.assertTrue(type(corr_mat(self.data_corr_df)), type(pd.DataFrame))
        self.assertTrue(type(corr_mat(self.data_corr_list)), type(pd.DataFrame))

    def test_output_shape(self):
        # Test for output of equal dimensions
        self.assertEqual(corr_mat(self.data_corr_df).data.shape[0], corr_mat(self.data_corr_df).data.shape[1])
        self.assertEqual(corr_mat(self.data_corr_list).data.shape[0], corr_mat(self.data_corr_list).data.shape[1])
