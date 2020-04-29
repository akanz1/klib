'''
Functions for data cleaning.

:author: Andreas Kanz

'''

# Imports
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .describe import corr_mat
from .utils import (_diff_report,
                    _drop_duplicates,
                    _missing_vals,
                    _validate_input_bool,
                    _validate_input_range)


__all__ = ['convert_datatypes',
           'data_cleaning',
           'drop_missing',
           'mv_col_handling']


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


def mv_col_handling(data, target=None, mv_threshold=0.1, corr_thresh_features=0.6, corr_thresh_target=0.3):
    '''
    Converts columns with a high ratio of missing values into binary features and eventually drops them based on \
    their correlation with other features and the target variable. This function follows a three step process:
    - 1) Identify features with a high ratio of missing values
    - 2) Identify high correlations of these features among themselves and with other features in the dataset.
    - 3) Features with high ratio of missing values and high correlation among each other are dropped unless \
         they correlate reasonably well with the target variable.

    Note: If no target is provided, the process exits after step two and drops columns identified up to this point.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    target: string, list, np.array or pd.Series, default None
        Specify target for correlation. E.g. label column to generate only the correlations between each feature \
        and the label.

    mv_threshold: float, default 0.1
        Value between 0 <= threshold <= 1. Features with a missing-value-ratio larger than mv_threshold are candidates \
        for dropping and undergo further analysis.

    corr_thresh_features: float, default 0.6
        Value between 0 <= threshold <= 1. Maximum correlation a previously identified features with a high mv-ratio is\
         allowed to have with another feature. If this threshold is overstepped, the feature undergoes further analysis.

    corr_thresh_target: float, default 0.3
        Value between 0 <= threshold <= 1. Minimum required correlation of a remaining feature (i.e. feature with a \
        high mv-ratio and high correlation to another existing feature) with the target. If this threshold is not met \
        the feature is ultimately dropped.

    Returns
    -------
    data: Updated Pandas DataFrame
    cols_mv: Columns with missing values included in the analysis
    drop_cols: List of dropped columns
    '''

    # Validate Inputs
    _validate_input_range(mv_threshold, 'mv_threshold', 0, 1)
    _validate_input_range(corr_thresh_features, 'corr_thresh_features', 0, 1)
    _validate_input_range(corr_thresh_target, 'corr_thresh_target', 0, 1)

    data = pd.DataFrame(data).copy()
    data_local = data.copy()
    mv_ratios = _missing_vals(data_local)['mv_cols_ratio']
    cols_mv = mv_ratios[mv_ratios > mv_threshold].index.tolist()
    data_local[cols_mv] = data_local[cols_mv].applymap(lambda x: 1 if not pd.isnull(x) else x).fillna(0)

    high_corr_features = []
    data_temp = data_local.copy()
    for col in cols_mv:
        corrmat = corr_mat(data_temp, colored=False)
        if abs(corrmat[col]).nlargest(2)[1] > corr_thresh_features:
            high_corr_features.append(col)
            data_temp = data_temp.drop(columns=[col])

    drop_cols = []
    if target is None:
        data = data.drop(columns=high_corr_features)
    else:
        for col in high_corr_features:
            if pd.DataFrame(data_local[col]).corrwith(target)[0] < corr_thresh_target:
                drop_cols.append(col)
                data = data.drop(columns=[col])

    return data, cols_mv, drop_cols


class MVColHandler(BaseEstimator, TransformerMixin):
    '''possible component of a cleaning pipeline --> follows DataCleaning'''

    def __init__(self, target=None, mch_mv_thresh=0.1, mch_feature_thresh=0.6, mch_target_thresh=0.3):
        self.target = target
        self.mch_mv_thresh = mch_mv_thresh
        self.mch_feature_thresh = mch_feature_thresh
        self.mch_target_thresh = mch_target_thresh

    def fit(self, data, target=None):
        return self

    def transform(self, data, target=None):
        data, cols_mv, dropped_cols = mv_col_handling(data, target=self.target, mv_threshold=self.mch_mv_thresh,
                                                      corr_thresh_features=self.mch_feature_thresh,
                                                      corr_thresh_target=self.mch_target_thresh)

        print(f'\nFeatures with MV-ratio > {self.mch_mv_thresh}: {len(cols_mv)}')
        print('Features dropped:', len(dropped_cols), dropped_cols)

        return data
