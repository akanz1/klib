from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from klib.describe import cat_plot
from klib.describe import corr_interactive_plot
from klib.describe import corr_mat
from klib.describe import corr_plot
from klib.describe import dist_plot
from klib.describe import missingval_plot


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
        assert corr_mat(self.data_corr_df).data.shape[0] == corr_mat(self.data_corr_df).data.shape[1]
        assert corr_mat(self.data_corr_list).data.shape[0] == corr_mat(self.data_corr_list).data.shape[1]
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


class Test_plots(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_cat_plot_smoke(self):
        data = pd.DataFrame(
            {
                "cat1": ["a", "b", "a", "c", "a", "b"],
                "cat2": ["x", "x", "y", "z", "z", "z"],
            },
        )

        assert cat_plot(data, figsize=(4, 4)) is not None

    def test_corr_plot_smoke(self):
        data = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],
                "c": [5, 4, 3, 2, 1],
            },
        )

        assert corr_plot(data, figsize=(4, 4)) is not None

    def test_corr_interactive_plot_smoke(self):
        data = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],
                "c": [5, 4, 3, 2, 1],
            },
        )

        assert corr_interactive_plot(data, figsize=(4, 4)) is not None

    def test_dist_plot_smoke(self):
        data = pd.DataFrame({"a": np.arange(30), "b": np.arange(30) ** 2})

        assert dist_plot(data, size=2) is not None

    def test_missingval_plot_smoke(self):
        data = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]})

        assert missingval_plot(data, figsize=(4, 4)) is not None
