'''
Functions for data cleaning.

:author: Andreas Kanz

'''

# Imports
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# from .preprocess import mv_col_handler
from .utils import _diff_report
from .utils import _drop_duplicates
from .utils import _missing_vals
from .utils import _validate_input_range
from .utils import _validate_input_bool


def convert_datatypes(data, category=True, cat_threshold=0.05, cat_exclude=None):
    '''
    Converts columns to best possible dtypes using dtypes supporting pd.NA. Temporarily not converting integers.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    category: bool, default True
        Change dtypes of columns with dtype "object" to "category". Set threshold using cat_threshold or exclude \
        columns using cat_exclude.

    cat_threshold: float, default 0.05
        Ratio of unique values below which categories are inferred and column dtype is changed to categorical.

    cat_exclude: list, default None
        List of columns to exclude from categorical conversion.

    Returns
    -------
    data: Pandas DataFrame
    '''

    # Validate Inputs
    _validate_input_bool(category, 'Category')
    _validate_input_range(cat_threshold, 'cat_threshold', 0, 1)

    cat_exclude = [] if cat_exclude is None else cat_exclude.copy()

    data = pd.DataFrame(data).copy()
    for col in data.columns:
        unique_vals_ratio = data[col].nunique(dropna=False) / data.shape[0]
        if (category and
            unique_vals_ratio < cat_threshold and
            col not in cat_exclude and
                data[col].dtype == 'object'):
            data[col] = data[col].astype('category')
        data[col] = data[col].convert_dtypes(infer_objects=True, convert_string=True,
                                             convert_integer=False, convert_boolean=True)

    return data


def drop_missing(data, drop_threshold_cols=1, drop_threshold_rows=1):
    '''
    Drops completely empty columns and rows by default and optionally provides flexibility to loosen restrictions to \
    drop additional columns and rows based on the fraction of remaining NA-values.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    drop_threshold_cols: float, default 1
        Drop columns with NA-ratio above the specified threshold.

    drop_threshold_rows: float, default 1
        Drop rows with NA-ratio above the specified threshold.

    Returns
    -------
    data_cleaned: Pandas DataFrame

    Notes
    -----
    Columns are dropped first. Rows are dropped based on the remaining data.
    '''

    # Validate Inputs
    _validate_input_range(drop_threshold_cols, 'drop_threshold_cols', 0, 1)
    _validate_input_range(drop_threshold_rows, 'drop_threshold_rows', 0, 1)

    data = pd.DataFrame(data).copy()
    data = data.dropna(axis=0, how='all').dropna(axis=1, how='all')
    data = data.drop(columns=data.loc[:, _missing_vals(data)['mv_cols_ratio'] > drop_threshold_cols].columns)
    data_cleaned = data.drop(index=data.loc[_missing_vals(data)['mv_rows_ratio'] > drop_threshold_rows, :].index)

    return data_cleaned


def data_cleaning(data, drop_threshold_cols=0.9, drop_threshold_rows=0.9, drop_duplicates=True,
                  convert_dtypes=True, category=True, cat_threshold=0.03, cat_exclude=None, show='changes'):
    '''
    Perform initial data cleaning tasks on a dataset, such as dropping single valued and empty rows, empty \
        columns as well as optimizing the datatypes.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    drop_threshold_cols: float, default 0.9
        Drop columns with NA-ratio above the specified threshold.

    drop_threshold_rows: float, default 0.9
        Drop rows with NA-ratio above the specified threshold.

    drop_duplicates: bool, default True
        Drop duplicate rows, keeping the first occurence. This step comes after the dropping of missing values.

    convert_dtypes: bool, default True
        Convert dtypes using pd.convert_dtypes().

    category: bool, default True
        Change dtypes of columns to "category". Set threshold using cat_threshold. Requires convert_dtypes=True

    cat_threshold: float, default 0.03
        Ratio of unique values below which categories are inferred and column dtype is changed to categorical.

    cat_exclude: list, default None
        List of columns to exclude from categorical conversion.

    show: {'all', 'changes', None} default 'all'
        Specify verbosity of the output.
        * 'all': Print information about the data before and after cleaning as well as information about changes.
        * 'changes': Print out differences in the data before and after cleaning.
        * None: No information about the data and the data cleaning is printed.

    Returns
    -------
    data_cleaned: Pandas DataFrame

    See Also
    --------
    convert_datatypes: Convert columns to best possible dtypes.
    drop_missing : Flexibly drop columns and rows.
    _memory_usage: Gives the total memory usage in kilobytes.
    _missing_vals: Metrics about missing values in the dataset.

    Notes
    -----
    The category dtype is not grouped in the summary, unless it contains exactly the same categories.
    '''

    # Validate Inputs
    _validate_input_range(drop_threshold_cols, 'drop_threshold_cols', 0, 1)
    _validate_input_range(drop_threshold_rows, 'drop_threshold_rows', 0, 1)
    _validate_input_bool(drop_duplicates, 'drop_duplicates')
    _validate_input_bool(convert_dtypes, 'convert_datatypes')
    _validate_input_bool(category, 'category')
    _validate_input_range(cat_threshold, 'cat_threshold', 0, 1)

    data = pd.DataFrame(data).copy()
    data_cleaned = drop_missing(data, drop_threshold_cols, drop_threshold_rows)

    single_val_cols = data_cleaned.columns[data_cleaned.nunique(dropna=False) == 1].tolist()
    data_cleaned = data_cleaned.drop(columns=single_val_cols)

    if drop_duplicates:
        data_cleaned, dupl_rows = _drop_duplicates(data_cleaned)
    if convert_dtypes:
        data_cleaned = convert_datatypes(data_cleaned, category=category, cat_threshold=cat_threshold,
                                         cat_exclude=cat_exclude)

    _diff_report(data, data_cleaned, dupl_rows=dupl_rows, single_val_cols=single_val_cols, show=show)

    return data_cleaned


class DataCleaner(BaseEstimator, TransformerMixin):
    '''Docstring of a class? methods also have docstrings or commments?'''
    '''possible component of a cleaning pipeline --> e.g. followed by MCH'''

    def __init__(self, drop_threshold_cols=0.9, drop_threshold_rows=0.9, drop_duplicates=True, convert_dtypes=True,
                 category=True, cat_threshold=0.03, cat_exclude=None, show='changes'):
        self.drop_threshold_cols = drop_threshold_cols
        self.drop_threshold_rows = drop_threshold_rows
        self.drop_duplicates = drop_duplicates
        self.convert_dtypes = convert_dtypes
        self.category = category
        self.cat_threshold = cat_threshold
        self.cat_exclude = cat_exclude
        self.show = show

    def fit(self, data, target=None):
        return self

    def transform(self, data, target=None):
        data_cleaned = data_cleaning(data, drop_threshold_cols=self.drop_threshold_cols,
                                     drop_threshold_rows=self.drop_threshold_rows, drop_duplicates=self.drop_duplicates,
                                     convert_dtypes=self.convert_dtypes, category=self.category, cat_threshold=self.
                                     cat_threshold, cat_exclude=self.cat_exclude, show=self.show)
        return data_cleaned
