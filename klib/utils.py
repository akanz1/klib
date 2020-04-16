'''
Utilities and auxiliary functions.

:author: Andreas Kanz

'''

# Imports
import numpy as np
import pandas as pd


def _corr_selector(corr, split=None, threshold=0):
    '''
    Parameters
    ----------
    corr: List or matrix of correlations.

    split: {None, 'pos', 'neg', 'high', 'low'}, default None
        Type of split to be performed.

    threshold: float, default 0
        Value between 0 <= threshold <= 1

    Returns:
    -------
    corr: List or matrix of (filtered) correlations.
    '''
    if split == 'pos':
        corr = corr.where((corr >= threshold) & (corr > 0))
        print('Displaying positive correlations. Use "threshold" to further limit the results.')
    elif split == 'neg':
        corr = corr.where((corr <= threshold) & (corr < 0))
        print('Displaying negative correlations. Use "threshold" to further limit the results.')
    elif split == 'high':
        corr = corr.where(np.abs(corr) >= threshold)
        print('Displaying absolute correlations above a chosen threshold.')
    elif split == 'low':
        corr = corr.where(np.abs(corr) <= threshold)
        print('Displaying absolute correlations below a chosen threshold.')
    else:
        corr = corr

    return corr


def _drop_duplicates(data):
    '''
    Provides information and drops duplicate rows.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    Returns
    -------
    data: Deduplicated Pandas DataFrame
    rows_dropped: Index Object of rows dropped.
    '''

    data = pd.DataFrame(data).copy()
    rows_dropped = data[data.duplicated()].index
    data = data.drop_duplicates()

    return data, rows_dropped


def _memory_usage(data):
    '''
    Gives the total memory usage in kilobytes.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    Returns
    -------
    memory_usage: float
    '''

    data = pd.DataFrame(data).copy()
    memory_usage = round(data.memory_usage(index=True, deep=True).sum()/1024, 2)

    return memory_usage


def _missing_vals(data):
    '''
    Gives metrics of missing values in the dataset.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    Returns
    -------
    mv_total: float, number of missing values in the entire dataset
    mv_rows: float, number of missing values in each row
    mv_cols: float, number of missing values in each column
    mv_rows_ratio: float, ratio of missing values for each row
    mv_cols_ratio: float, ratio of missing values for each column
    '''

    data = pd.DataFrame(data).copy()
    mv_rows = data.isna().sum(axis=1)
    mv_cols = data.isna().sum(axis=0)
    mv_total = data.isna().sum().sum()
    mv_rows_ratio = mv_rows/data.shape[1]
    mv_cols_ratio = mv_cols/data.shape[0]

    return {'mv_total': mv_total,
            'mv_rows': mv_rows,
            'mv_cols': mv_cols,
            'mv_rows_ratio': mv_rows_ratio,
            'mv_cols_ratio': mv_cols_ratio}


def _validate_input_0_1(value, desc):
    if value < 0 or value > 1:
        raise ValueError(f'Input value for {desc} is {value} but should be a float in the range 0 <= {desc} <=1.')


def _validate_input_bool(value, desc):
    if not(isinstance(value, bool)):
        raise ValueError(f'Input value for {desc} is {value} but should be boolean.')
