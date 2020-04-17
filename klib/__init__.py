"""
Data Science Module for Python
==================================
klib is an easy to use Python library of customized functions for cleaning and analyzing data.
"""

__author__ = """Andreas Kanz"""

from ._version import __version__
from .describe import corr_mat, corr_plot, dist_plot, missingval_plot
from .clean import convert_datatypes, data_cleaning, drop_missing

__version__ = __version__

__all__ = ['convert_datatypes',
           'corr_mat',
           'corr_plot',
           'data_cleaning',
           'dist_plot',
           'drop_missing',
           'missingval_plot']
