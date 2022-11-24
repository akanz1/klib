"""
Data Science Module for Python
===============================
klib is an easy to use Python library of customized functions for cleaning and \
analyzing data.
"""

__author__ = """Andreas Kanz"""

from klib._version import __version__
from klib.clean import clean_column_names
from klib.clean import convert_datatypes
from klib.clean import data_cleaning
from klib.clean import drop_missing
from klib.clean import mv_col_handling
from klib.clean import pool_duplicate_subsets
from klib.describe import cat_plot
from klib.describe import corr_mat
from klib.describe import corr_plot
from klib.describe import dist_plot
from klib.describe import missingval_plot

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
