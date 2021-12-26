"""
Functions for data preprocessing.

:author: Andreas Kanz

"""

# Imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, RobustScaler

from klib.utils import (
    _validate_input_int,
    _validate_input_range,
    _validate_input_sum_smaller,
)

__all__ = ["feature_selection_pipe", "num_pipe", "cat_pipe", "train_dev_test_split"]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Determines and selects numerical and categorical columns from a dataset based on \
    their supposed dtype. Unlike sklearn's make_column_selector() missing values are \
    temporarily filled in to allow convert_dtypes() to determine the dtype of a column.

    Parameter
    ---------
    num: default, True
        Select only numerical Columns. If num = False, only categorical columns are \
        selected.

    Returns
    -------
    Dataset containing only numerical or categorical data.
    """

    def __init__(self, num=True):
        self.num = num

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        temp = X.fillna(X.mode().iloc[0]).convert_dtypes()

        if self.num:
            return X[temp.select_dtypes(include=["number"]).columns.tolist()]
        return X[temp.select_dtypes(exclude=["number"]).columns.tolist()]


class PipeInfo(BaseEstimator, TransformerMixin):
    """
    Prints intermediary information about the dataset from within a pipeline.

    Include at any point in a Pipeline to print out the shape of the dataset at this \
    point and to receive an indication of the progress within the pipeline.

    Set to 'None' to avoid printing the shape of the dataset. This parameter can also \
    be set as a hyperparameter, e.g. 'pipeline__pipeinfo-1': [None] or \
    'pipeline__pipeinfo-1__name': ['my_custom_name'].

    Parameter
    ---------
    name: string, default None
        Provide a name for the current step.

    Returns
    -------
    Data: Data is being passed through.
    """

    def __init__(self, name=None):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"Step: {self.name} --- Shape: {X.shape}")
        return X


def cat_pipe(
    imputer=SimpleImputer(strategy="most_frequent"),
    encoder=OneHotEncoder(handle_unknown="ignore"),
    scaler=MaxAbsScaler(),
    encoder_info=PipeInfo(name="after encoding categorical data"),
):
    """
    Standard preprocessing operations on categorical data.

    Parameters
    ----------
    imputer: default, SimpleImputer(strategy='most_frequent')

    encoder: default, OneHotEncoder(handle_unknown='ignore')
        Encode categorical features as a one-hot numeric array.

    scaler: default, MaxAbsScaler()
        Scale each feature by its maximum absolute value. MaxAbsScaler() does not \
        shift/center the data, and thus does not destroy any sparsity. It is \
        recommended to check for outliers before applying MaxAbsScaler().

    encoder_info:
        Prints the shape of the dataset at the end of 'cat_pipe'. Set to 'None' to \
        avoid printing the shape of dataset. This parameter can also be set as a \
        hyperparameter, e.g. 'pipeline__pipeinfo-1': [None] or \
        'pipeline__pipeinfo-1__name': ['my_custom_name'].

    Returns
    -------
    Pipeline
    """
    return make_pipeline(
        ColumnSelector(num=False), imputer, encoder, encoder_info, scaler
    )


def feature_selection_pipe(
    var_thresh=VarianceThreshold(threshold=0.1),
    select_from_model=SelectFromModel(
        LassoCV(cv=4, random_state=408), threshold="0.1*median"
    ),
    select_percentile=SelectPercentile(f_classif, percentile=95),
    var_thresh_info=PipeInfo(name="after var_thresh"),
    select_from_model_info=PipeInfo(name="after select_from_model"),
    select_percentile_info=PipeInfo(name="after select_percentile"),
):
    """
    Preprocessing operations for feature selection.

    Parameters
    ----------
    var_thresh: default, VarianceThreshold(threshold=0.1)
        Specify a threshold to drop low variance features.

    select_from_model: default, SelectFromModel(LassoCV(cv=4, random_state=408), \
    threshold="0.1 * median")
        Specify an estimator which is used for selecting features based on importance \
        weights.

    select_percentile: default, SelectPercentile(f_classif, percentile=95)
        Specify a score-function and a percentile value of features to keep.

    var_thresh_info, select_from_model_info, select_percentile_info
        Prints the shape of the dataset after applying the respective function. Set to \
        'None' to avoid printing the shape of dataset. This parameter can also be set \
        as a hyperparameter, e.g. 'pipeline__pipeinfo-1': [None] \
        or 'pipeline__pipeinfo-1__name': ['my_custom_name'].

    Returns
    -------
    Pipeline
    """
    return make_pipeline(
        var_thresh,
        var_thresh_info,
        select_from_model,
        select_from_model_info,
        select_percentile,
        select_percentile_info,
    )


def num_pipe(
    imputer=IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=25, n_jobs=4, random_state=408),
        random_state=408,
    ),
    scaler=RobustScaler(),
):
    """
    Standard preprocessing operations on numerical data.

    Parameters
    ----------
    imputer: default, IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=25, \
    n_jobs=4, random_state=408), random_state=408)

    scaler: default, RobustScaler()

    Returns
    -------
    Pipeline
    """
    return make_pipeline(ColumnSelector(), imputer, scaler)


def train_dev_test_split(
    data, target, dev_size=0.1, test_size=0.1, stratify=None, random_state=408
):
    """
    Split a dataset and a label column into train, dev and test sets.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
    is provided, the index/column information is used to label the plots.

    target: string, list, np.array or pd.Series, default None
        Specify target for correlation. E.g. label column to generate only the \
        correlations between each feature and the label.

    dev_size: float, default 0.1
        If float, should be between 0.0 and 1.0 and represent the proportion of the \
        dataset to include in the dev split.

    test_size: float, default 0.1
        If float, should be between 0.0 and 1.0 and represent the proportion of the \
        dataset to include in the test split.

    stratify: target column, default None
        If not None, data is split in a stratified fashion, using the input as the \
        class labels.

    random_state: integer, default 408
        Random_state is the seed used by the random number generator.

    Returns
    -------
    tuple: Tuple containing train-dev-test split of inputs.
    """
    # Validate Inputs
    _validate_input_range(dev_size, "dev_size", 0, 1)
    _validate_input_range(test_size, "test_size", 0, 1)
    _validate_input_int(random_state, "random_state")
    _validate_input_sum_smaller(1, "Dev and test", dev_size, test_size)

    target_data = []
    if isinstance(target, str):
        target_data = data[target]
        data = data.drop(target, axis=1)

    elif isinstance(target, (list, pd.Series, np.ndarray)):
        target_data = pd.Series(target)

    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data,
        target_data,
        test_size=dev_size + test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if (dev_size == 0) or (test_size == 0):
        return X_train, X_dev_test, y_train, y_dev_test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_dev_test,
        y_dev_test,
        test_size=test_size / (dev_size + test_size),
        random_state=random_state,
        stratify=y_dev_test,
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test
