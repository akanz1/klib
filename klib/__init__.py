"""
Data Science Module for Python
==================================
klib is an easy to use Python library of customized functions for cleaning and analyzing data.
"""

__author__ = """Andreas Kanz"""

from ._version import __version__
from .clean import (convert_datatypes,
                    data_cleaning,
                    drop_missing,
                    mv_col_handling,
                    pool_duplicate_subsets)
from .describe import (cat_plot,
                       corr_mat,
                       corr_plot,
                       dist_plot,
                       missingval_plot)
from .preprocess import (feature_selection_pipe,
                         num_pipe,
                         cat_pipe,
                         train_dev_test_split)

__version__ = __version__

__all__ = ['convert_datatypes',
           'data_cleaning',
           'drop_missing',
           'mv_col_handling',
           'pool_duplicate_subsets',
           'cat_plot',
           'corr_mat',
           'corr_plot',
           'dist_plot',
           'missingval_plot',
           'feature_selection_pipe',
           'num_pipe',
           'cat_pipe',
           'train_dev_test_split',
           '__version__']

# in future versions and especially with an increased number of functions, only the most frequently used functions will\
# be imported into the namespace to be accessible from klib.function directly. The remaining functions can be found in \
# the respective modules:
#     'clean',
#    'describe',
#    'preprocess',
