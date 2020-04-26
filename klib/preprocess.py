'''
Functions for data preprocessing.

:author: Andreas Kanz

'''

# Imports
import pandas as pd

from .describe import corr_mat
from .utils import _missing_vals
from .utils import _validate_input_range


def mv_col_handler(data, target=None, mv_threshold=0.25, corr_thresh_features=0.65, corr_thresh_target=0.2):
    '''
    Drops columns with a high ratio of missing values based on correlation with other features and the target variable.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    target: string, list, np.array or pd.Series, default None
        Specify target for correlation. E.g. label column to generate only the correlations between each feature \
        and the label.

    mv_threshold: float, default 0.25
        Value between 0 <= threshold <= 1. Features with a missing-value-ratio larger than mv_threshold are candidates \
        for dropping and undergo further analysis.

    corr_thresh_features: float, default 0.65
        Value between 0 <= threshold <= 1. Previously identified features with a high mv-ratio with a correlation \
        larger than corr_thresh_features with any other feature undergo further analysis.

    corr_thresh_target: float, default 0.25
        Value between 0 <= threshold <= 1. The remaining features (with a high mv-ratio and high correlation to an \
        existing feature) are dropped unless their correlation with the target is larger than corr_thresh_target.

    Returns
    -------
    data: Updated Pandas DataFrame
    drop_cols: List of dropped columns
    '''

    # Validate Inputs
    _validate_input_range(mv_threshold, 'mv_threshold', -1, 1)
    _validate_input_range(corr_thresh_features, 'corr_thresh_features', -1, 1)
    _validate_input_range(corr_thresh_target, 'corr_thresh_target', -1, 1)

    data = pd.DataFrame(data).copy()
    mv_ratios = _missing_vals(data)['mv_cols_ratio']
    cols_mv = mv_ratios[mv_ratios > mv_threshold].index.tolist()
    data_mv_binary = data[cols_mv].applymap(lambda x: 1 if not pd.isnull(x) else x).fillna(0)

    for col in cols_mv:
        data[col] = data_mv_binary[col]

    high_corr_features = []
    data_temp = data.copy()
    for col in cols_mv:
        corrmat = corr_mat(data_temp, colored=False)
        if abs(corrmat[col]).nlargest(2)[1] > corr_thresh_features:
            high_corr_features.append(col)
            data_temp = data_temp.drop(columns=[col])

    drop_cols = []
    for col in high_corr_features:
        if pd.DataFrame(data_mv_binary[col]).corrwith(target)[0] < corr_thresh_target:
            drop_cols.append(col)
            data = data.drop(columns=[col])

    return data, drop_cols
