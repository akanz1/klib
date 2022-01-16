"""
Data Science Module for Python
==================================
klib is an easy to use Python library of customized functions for cleaning and \
analyzing data.
"""

__author__ = """Andreas Kanz"""

from klib._version import __version__
from klib.clean import (
    clean_column_names,
    convert_datatypes,
    data_cleaning,
    drop_missing,
    mv_col_handling,
    pool_duplicate_subsets,
)
from klib.describe import cat_plot, corr_mat, corr_plot, dist_plot, missingval_plot

__all__ = [
    "clean_column_names",
    "convert_datatypes",
    "data_cleaning",
    "drop_missing",
    "mv_col_handling",
    "pool_duplicate_subsets",
    "cat_plot",
    "corr_mat",
    "corr_plot",
    "dist_plot",
    "missingval_plot",
    "__version__",
]
