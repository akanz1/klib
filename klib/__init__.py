# flake8: noqa

"""
Data Science Module for Python
==================================
klib is an easy to use Python library of customized functions for cleaning and analyzing data.
"""

__author__ = """Andreas Kanz"""

from ._version import __version__
from .clean import *
from .describe import *
from .preprocess import *

__version__ = __version__

# __all__ = ['cat_plot',
#            'convert_datatypes',
#            'corr_mat',
#            'corr_plot',
#            'data_cleaning',
#            'dist_plot',
#            'drop_missing',
#            'missingval_plot',
#            'train_dev_test_split']
