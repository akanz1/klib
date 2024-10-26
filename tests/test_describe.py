from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from klib.describe import corr_mat


class Test_corr_mat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_corr_df = pd.DataFrame(
            [[1, 0, 3, 4, "a"], [3, 4, 5, 6, "b"], [5, 4, 2, 1, "c"]],
            columns=["Col1", "Col2", "Col3", "Col4", "Col5"],
        )

        cls.data_corr_list = [1, 2, -3]
        cls.data_corr_target_series = pd.Series([1, 2, -3], name="Target Series")
        cls.data_corr_target_array = np.array([1, 2, -3])
        cls.data_corr_target_list = [1, 2, -3]

    def test_output_type(self):
        # Test conversion from pd.io.formats.style.Styler to pd.core.frame.DataFrame
        assert isinstance(
            type(corr_mat(self.data_corr_df)),
            type(pd.io.formats.style.Styler),
        )
        assert isinstance(
            type(corr_mat(self.data_corr_list)),
            type(pd.io.formats.style.Styler),
        )
        assert isinstance(
            type(corr_mat(self.data_corr_df, target="Col1")),
            type(pd.io.formats.style.Styler),
        )
        assert isinstance(
            type(corr_mat(self.data_corr_df, target=self.data_corr_target_series)),
            type(pd.io.formats.style.Styler),
        )
        assert isinstance(
            type(corr_mat(self.data_corr_df, target=self.data_corr_target_array)),
            type(pd.io.formats.style.Styler),
        )
        assert isinstance(
            type(corr_mat(self.data_corr_df, target=self.data_corr_target_list)),
            type(pd.io.formats.style.Styler),
        )

        assert isinstance(
            type(corr_mat(self.data_corr_df, colored=False)),
            type(pd.DataFrame),
        )
        assert isinstance(
            type(corr_mat(self.data_corr_list, colored=False)),
            type(pd.DataFrame),
        )
        assert isinstance(
            type(corr_mat(self.data_corr_df, target="Col1", colored=False)),
            type(pd.DataFrame),
        )
        assert isinstance(
            type(
                corr_mat(
                    self.data_corr_df,
                    target=self.data_corr_target_series,
                    colored=False,
                ),
            ),
            type(pd.DataFrame),
        )
        assert isinstance(
            type(
                corr_mat(
                    self.data_corr_df,
                    target=self.data_corr_target_array,
                    colored=False,
                ),
            ),
            type(pd.DataFrame),
        )
        assert isinstance(
            type(
                corr_mat(
                    self.data_corr_df,
                    target=self.data_corr_target_list,
                    colored=False,
                ),
            ),
            type(pd.DataFrame),
        )

    def test_output_shape(self):
        # Test for output dimensions
        assert (
            corr_mat(self.data_corr_df).data.shape[0] == corr_mat(self.data_corr_df).data.shape[1]
        )
        assert (
            corr_mat(self.data_corr_list).data.shape[0]
            == corr_mat(self.data_corr_list).data.shape[1]
        )
        assert corr_mat(self.data_corr_df, target="Col1", colored=False).shape == (3, 1)
        assert corr_mat(
            self.data_corr_df,
            target=self.data_corr_target_series,
            colored=False,
        ).shape == (4, 1)
        assert corr_mat(
            self.data_corr_df,
            target=self.data_corr_target_array,
            colored=False,
        ).shape == (4, 1)
        assert corr_mat(
            self.data_corr_df,
            target=self.data_corr_target_list,
            colored=False,
        ).shape == (4, 1)
