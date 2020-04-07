'''
Utilities for data cleaning.

:author: Andreas Kanz

'''

# Imports
import pandas as pd

from .describe import _memory_usage
from .describe import _missing_vals


def convert_datatypes(data, category=True, cat_threshold=0.05, cat_exclude=[]):
    '''
    Converts columns to best possible dtypes using dtypes supporting pd.NA.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame is provided, the index/column \
    information is used to label the plots.

    category: bool, default True
        Change dtypes of columns to "category". Set threshold using cat_threshold.

    cat_threshold: float, default 0.05
        Ratio of unique values below which categories are inferred and column dtype is changed to categorical.

    cat_exclude: default [] (empty list)
        List of columns to exclude from categorical conversion.

    Returns
    -------
    Pandas DataFrame.

    '''

    data = pd.DataFrame(data).copy()
    for col in data.columns:
        unique_vals_ratio = data[col].nunique(dropna=False) / data.shape[0]
        if (category and
            unique_vals_ratio < cat_threshold and
            col not in cat_exclude and
                data[col].dtype == 'object'):
            data[col] = data[col].astype('category')
        data[col] = data[col].convert_dtypes()

    return data


def drop_missing(data, drop_threshold_cols=1, drop_threshold_rows=1):
    '''
    Drops entirely empty columns and rows by default and optionally provides flexibility to loosen restrictions to \
    drop additional columns and rows based on the fraction of NA-values.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame is provided, the index/column \
    information is used to label the plots.

    drop_threshold_cols: float, default 1
        Drop columns with NA-ratio above the specified threshold.

    drop_threshold_rows: float, default 1
        Drop rows with NA-ratio above the specified threshold.

    Returns
    -------
    Pandas DataFrame.

    Notes
    -----
    Columns are dropped first. Rows are dropped based on the remaining data.

    '''

    data = pd.DataFrame(data)
    data = data.dropna(axis=0, how='all')
    data = data.dropna(axis=1, how='all')
    data = data.drop(columns=data.loc[:, _missing_vals(data)['mv_cols_ratio'] > drop_threshold_cols].columns)
    data_cleaned = data.drop(index=data.loc[_missing_vals(data)['mv_rows_ratio'] > drop_threshold_rows, :].index)

    return data_cleaned


def data_cleaning(data, drop_threshold_cols=0.95, drop_threshold_rows=0.95, category=True,
                  cat_threshold=0.03, cat_exclude=[], show='changes'):
    '''
    Perform initial data cleaning tasks on a dataset, such as dropping empty rows and columns and optimizing the \
    datatypes.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame is provided, the index/column \
    information is used to label the plots.

    drop_threshold_cols: float, default 0.95
    Drop columns with NA-ratio above the specified threshold.

    drop_threshold_rows: float, default 0.95
    Drop rows with NA-ratio above the specified threshold.

    category: bool, default True
        Change dtypes of columns to "category". Set threshold using cat_threshold.

    cat_threshold: float, default 0.03
        Ratio of unique values below which categories are inferred and column dtype is changed to categorical.

    cat_exclude: default [] (empty list)
        List of columns to exclude from categorical conversion.

    show: {'all', 'changes', None} default 'all'
        Specify verbosity of the output.
        * 'all': Print information about the data before and after cleaning as well as information about changes.
        * 'changes': Print out differences in the data before and after cleaning.
        * None: no information about the data is printed.

    Returns
    -------
    Pandas DataFrame.

    See Also
    --------
    convert_datatypes: Converts columns to best possible dtypes.
    drop_missing : Flexibly drops columns and rows.
    _memory_usage: Gives the total memory usage in kilobytes.
    _missing_vals: Metrics about missing values in the dataset.


    Notes
    -----
    The category dtype is not grouped in the summary, unless it contains exactly the same categories.

    '''

    data = pd.DataFrame(data)
    data_cleaned = drop_missing(data, drop_threshold_cols, drop_threshold_rows)
    data_cleaned = convert_datatypes(data_cleaned, category=category, cat_threshold=cat_threshold,
                                     cat_exclude=cat_exclude)

    if show in ['changes', 'all']:
        if show == 'all':
            print('Before data cleaning:\n')
            print(f'dtypes:\n{data.dtypes.value_counts()}')
            print(f'\nNumber of rows: {data.shape[0]}')
            print(f'Number of cols: {data.shape[1]}')
            print(f"Missing values: {_missing_vals(data)['mv_total']}")
            print(f'Memory usage: {_memory_usage(data)} KB')
            print('_______________________________________________________\n')
            print('After data cleaning:\n')
            print(f'dtypes:\n{data_cleaned.dtypes.value_counts()}')
            print(f'\nNumber of rows: {data_cleaned.shape[0]}')
            print(f'Number of cols: {data_cleaned.shape[1]}')
            print(f"Missing values: {_missing_vals(data_cleaned)['mv_total']}")
            print(f'Memory usage: {_memory_usage(data_cleaned)} KB')
            print('_______________________________________________________\n')

        print(
            f"Shape of cleaned data: {data_cleaned.shape} - Remaining NAs: {_missing_vals(data_cleaned)['mv_total']}")
        print(f'\nChanges:')
        print(f'Dropped rows: {data.shape[0]-data_cleaned.shape[0]}')
        print(f'Dropped columns: {data.shape[1]-data_cleaned.shape[1]}')
        print(f"Dropped missing values: {_missing_vals(data)['mv_total']-_missing_vals(data_cleaned)['mv_total']}")
        mem_change = _memory_usage(data)-_memory_usage(data_cleaned)
        print(f'Reduced memory by: {round(mem_change,2)} KB (-{round(100*mem_change/_memory_usage(data),1)}%)')

    return data_cleaned
