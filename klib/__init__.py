"""
Data Science Module for Python
==================================
klib is an easy to use Python library of customized functions for cleaning and analyzing data.
"""

__author__ = """Andreas Kanz"""

from ._version import __version__
from .clean import convert_datatypes, data_cleaning, drop_missing
from .describe import cat_plot, corr_mat, corr_plot, dist_plot, missingval_plot
from .preprocess import train_dev_test_split

__version__ = __version__

__all__ = ['cat_plot',
           'convert_datatypes',
           'corr_mat',
           'corr_plot',
           'data_cleaning',
           'dist_plot',
           'drop_missing',
           'missingval_plot',
           'train_dev_test_split']
