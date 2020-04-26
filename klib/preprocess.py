'''
Functions for data preprocessing.

:author: Andreas Kanz

'''

# Imports
import pandas as pd

from .describe import corr_mat
from .utils import _missing_vals
from .utils import _validate_input_range


def mv_col_handler(data, target=None, mv_threshold=0.2, corr_thresh_features=0.6, corr_thresh_target=0.30):
    '''
    Drop columns with a high ratio of missing values based on correlation with other features and the target \
    variable. This function follows a three step process:
    - 1) Identify features with a high ratio of missing values
    - 2) Identify high correlations of these features among themselves and with other features in the dataset.
    - 3) Features with high ratio of missing values and high correlation among each other are dropped unless \
         they correlate reasonably well with the target variable.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame.

    target: string, list, np.array or pd.Series, default None
        Specify target for correlation. E.g. label column to generate only the correlations between each feature \
        and the label.

    mv_threshold: float, default 0.2
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
