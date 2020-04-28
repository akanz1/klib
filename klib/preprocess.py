'''
Functions for data preprocessing.

:author: Andreas Kanz

'''

# Imports
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .utils import _validate_input_int
from .utils import _validate_input_range


__all__ = ['train_dev_test_split',
           'cat_pipe',
           'num_pipe',
           'preprocessing_pipe']


def train_dev_test_split(data, target, dev_size=0.1, test_size=0.1, stratify=None, random_state=408):
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

    random_state: integer, default 408
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


class ColumnSelector(BaseEstimator, TransformerMixin):
    ''''''

    def __init__(self, num=True):
        self.num = num

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        temp = X.fillna(X.mode().iloc[0]).convert_dtypes()

        if self.num:
            return X[temp.select_dtypes(include=['number']).columns.tolist()]
        else:
            return X[temp.select_dtypes(exclude=['number']).columns.tolist()]


def num_pipe(imputer=IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=25, n_jobs=4, random_state=408), random_state=408),
        scaler=RobustScaler()):
    '''Set of standard preprocessing operations on numerical data.'''

    num_pipe = make_pipeline(ColumnSelector(),
                             (imputer),
                             (scaler))
    return num_pipe


def cat_pipe(imputer=SimpleImputer(strategy='most_frequent')):
    '''Set of standard preprocessing operations on categorical data.'''

    cat_pipe = make_pipeline(ColumnSelector(num=False),
                             imputer,
                             OneHotEncoder(handle_unknown='ignore'))
    return cat_pipe


def preprocessing_pipe(num=True, cat=True):
    '''Set of standard preprocessing operations on numerical and categorical data.

    Parameters:
    ----------
    num: bool, default True
        Set to false if no numerical data is in the dataset.

    cat: bool, default True
        Set to false if no categorical data is in the dataset.
    '''

    pipe = None
    if num and cat:
        pipe = make_union(num_pipe(), cat_pipe(), n_jobs=4)

    elif num:
        pipe = num_pipe()

    elif cat:
        pipe = cat_pipe()

    return pipe
