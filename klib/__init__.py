"""
Data Science Module for Python
==================================
klib is a Python module ... ...
"""

__author__ = """Andreas Kanz"""

from ._version import __version__
from .describe import corr_plot, missingval_plot
from .clean import convert_datatypes, data_cleaning, drop_missing

__version__ = __version__

__all__ = ['corr_plot', 'missingval_plot', 'convert_datatypes', 'data_cleaning', 'drop_missing']
