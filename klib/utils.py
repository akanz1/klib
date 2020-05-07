'''
Utilities and auxiliary functions.

:author: Andreas Kanz

'''

# Imports
import numpy as np
import pandas as pd


def _corr_selector(corr, split=None, threshold=0):
    '''
    Select correlations based on the provided parameters.

    Parameters
    ----------
    corr: pd.Series or pd.DataFrame of correlations.

    split: {None, 'pos', 'neg', 'above', 'below'}, default None
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
    elif split == 'above':
        corr = corr.where(np.abs(corr) >= threshold)
        print(f'Displaying absolute correlations above the threshold ({threshold}).')
    elif split == 'below':
        corr = corr.where(np.abs(corr) <= threshold)
        print(f'Displaying absolute correlations below the threshold ({threshold}).')

    return corr


def _diff_report(data, data_cleaned, dupl_rows=None, single_val_cols=None, show='changes'):
    '''
    Provides information about changes between two datasets, such as dropped rows and columns, memory usage and \
    missing values.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.
        Input the initial dataset here.

    data_cleaned: 2D dataset that can be coerced into Pandas DataFrame.
        Input the cleaned / updated dataset here.

    dupl_rows: list, default None
        List of duplicate row indices.

    single_val_cols: list, default None
        List of single-valued column indices. I.e. columns where all cells contain the same value. \
        NaNs count as a separate value.

    show: {'all', 'changes', None} default 'all'
        Specify verbosity of the output.
        * 'all': Print information about the data before and after cleaning as well as information about changes.
        * 'changes': Print out differences in the data before and after cleaning.
        * None: No information about the data and the data cleaning is printed.

    Returns:
    -------
    Print statement highlighting the datasets or changes between the two datasets.
    '''

    if show in ['changes', 'all']:
        dupl_rows = [] if dupl_rows is None else dupl_rows.copy()
        single_val_cols = [] if single_val_cols is None else single_val_cols.copy()
        data_mem = _memory_usage(data)
        data_cl_mem = _memory_usage(data_cleaned)
        data_mv_tot = _missing_vals(data)['mv_total']
        data_cl_mv_tot = _missing_vals(data_cleaned)['mv_total']

        if show == 'all':
            print('Before data cleaning:\n')
            print(f'dtypes:\n{data.dtypes.value_counts()}')
            print(f'\nNumber of rows: {data.shape[0]}')
            print(f'Number of cols: {data.shape[1]}')
            print(f'Missing values: {data_mv_tot}')
            print(f'Memory usage: {data_mem} KB')
            print('_______________________________________________________\n')
            print('After data cleaning:\n')
            print(f'dtypes:\n{data_cleaned.dtypes.value_counts()}')
            print(f'\nNumber of rows: {data_cleaned.shape[0]}')
            print(f'Number of cols: {data_cleaned.shape[1]}')
            print(f'Missing values: {data_cl_mv_tot}')
            print(f'Memory usage: {data_cl_mem} KB')
            print('_______________________________________________________\n')

        print(f'Shape of cleaned data: {data_cleaned.shape} - Remaining NAs: {data_cl_mv_tot}')
        print(f'\nChanges:')
        print(f'Dropped rows: {data.shape[0]-data_cleaned.shape[0]}')
        print(f'     of which {len(dupl_rows)} duplicates. (Rows: {dupl_rows})')
        print(f'Dropped columns: {data.shape[1]-data_cleaned.shape[1]}')
        print(f'     of which {len(single_val_cols)} single valued. (Columns: {single_val_cols})')
        print(f'Dropped missing values: {data_mv_tot-data_cl_mv_tot}')
        mem_change = data_mem-data_cl_mem
        print(f'Reduced memory by: {round(mem_change,2)} KB (-{round(100*mem_change/data_mem,1)}%)')


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
    dupl_rows = data[data.duplicated()].index.tolist()
    data = data.drop(dupl_rows, axis='index')

    return data, dupl_rows


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


def _validate_input_bool(value, desc):
    if not(isinstance(value, bool)):
        raise TypeError(f"Input value for '{desc}' is {type(value)} but should be a boolean.")


def _validate_input_int(value, desc):
    if type(value) != int:
        raise TypeError(f"Input value for '{desc}' is {type(value)} but should be an integer.")


def _validate_input_range(value, desc, lower, upper):
    if value < lower or value > upper:
        raise ValueError(
            f"'{desc}' = {value} but should be within the range {lower} <= '{desc}' <= {upper}.")


def _validate_input_smaller(value1, value2, desc):
    if value1 > value2:
        raise ValueError(f"The first input for '{desc}' should be smaller or equal to the second input.")


def _validate_input_sum(limit, desc, *args):
    if sum(args) > limit:
        raise ValueError(f"The sum of imput values provided for '{desc}' should be less or equal to {limit}.")
