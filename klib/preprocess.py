'''
Functions for data preprocessing.

:author: Andreas Kanz

'''

# Imports
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .describe import corr_mat
from .utils import _missing_vals
from .utils import _validate_input_int
from .utils import _validate_input_range


def mv_col_handler(data, target=None, mv_threshold=0.1, corr_thresh_features=0.6, corr_thresh_target=0.3):
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


def train_dev_test_split(data, target, dev_size=0.1, test_size=0.1, stratify=None, random_state=1234):
    '''
    Split a dataset and a label column into train, dev and test sets.

    Parameters:
    ----------

    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame is provided, the index/column \
    information is used to label the plots.

    target: string, list, np.array or pd.Series, default None
        Specify target for correlation. E.g. label column to generate only the correlations between each feature \
        and the label.

    dev_size: float, default 0.1
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the dev \
        split.

    test_size: float, default 0.1
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test \
        split.

    stratify: target column, default None
        If not None, data is split in a stratified fashion, using the input as the class labels.

    random_state: integer
        Random_state is the seed used by the random number generator.

    Returns
    -------
    tuple: Tuple containing train-dev-test split of inputs.
    '''

    # Validate Inputs
    _validate_input_range(dev_size, 'dev_size', 0, 1)
    _validate_input_range(test_size, 'test_size', 0, 1)
    _validate_input_int(random_state, 'random_state')

    target_data = []
    if isinstance(target, str):
        target_data = data[target]
        data = data.drop(target, axis=1)

    elif isinstance(target, (list, pd.Series, np.ndarray)):
        target_data = pd.Series(target)
        target = target.name

    X_train, X_dev_test, y_train, y_dev_test = train_test_split(data, target_data,
                                                                test_size=dev_size+test_size,
                                                                random_state=random_state,
                                                                stratify=stratify)

    if (dev_size == 0) or (test_size == 0):
        return X_train, X_dev_test, y_train, y_dev_test

    else:
        X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test,
                                                        test_size=test_size/(dev_size+test_size),
                                                        random_state=random_state,
                                                        stratify=y_dev_test)
        return X_train, X_dev, X_test, y_train, y_dev, y_test
