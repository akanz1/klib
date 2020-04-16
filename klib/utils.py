'''
Utilities and auxiliary functions.

:author: Andreas Kanz

'''

# Imports
import pandas as pd


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

    data = pd.DataFrame(data)
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

    data = pd.DataFrame(data)
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
