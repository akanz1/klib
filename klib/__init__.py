# flake8: noqa

"""
Data Science Module for Python
==================================
klib is an easy to use Python library of customized functions for cleaning and analyzing data.
"""

__author__ = """Andreas Kanz"""

from ._version import __version__
from . import clean
from . import describe
from . import preprocess

__version__ = __version__

__all__ = ['clean',
           'describe',
           'preprocess',
           '__version__']
