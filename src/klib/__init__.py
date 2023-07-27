"""Data cleaning and visualisation functions for Python.

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
from klib.describe import corr_interactive_plot
from klib.describe import corr_mat
from klib.describe import corr_plot
from klib.describe import dist_plot
from klib.describe import missingval_plot

__all__ = [
    "__version__",
    "cat_plot",
    "clean_column_names",
    "convert_datatypes",
    "corr_interactive_plot",
    "corr_mat",
    "corr_plot",
    "data_cleaning",
    "dist_plot",
    "drop_missing",
    "missingval_plot",
    "mv_col_handling",
    "pool_duplicate_subsets",
]
