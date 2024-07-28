"""Functions for data cleaning.

:author: Andreas Kanz
"""

from __future__ import annotations

import itertools
import re
from typing import Literal

import numpy as np
import pandas as pd

from klib.describe import corr_mat
from klib.utils import _diff_report
from klib.utils import _drop_duplicates
from klib.utils import _missing_vals
from klib.utils import _validate_input_bool
from klib.utils import _validate_input_range

__all__ = [
    "clean_column_names",
    "convert_datatypes",
    "data_cleaning",
    "drop_missing",
    "mv_col_handling",
]


def _optimize_ints(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(data).copy()
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def _optimize_floats(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(data).copy()
    floats = data.select_dtypes(include=["float64"]).columns.tolist()
    data[floats] = data[floats].apply(pd.to_numeric, downcast="float")
    return data


def clean_column_names(data: pd.DataFrame, *, hints: bool = True) -> pd.DataFrame:
    """Clean the column names of the provided Pandas Dataframe.

    Optionally provides hints on duplicate and long column names.

    Parameters
    ----------
    data : pd.DataFrame
        Original Dataframe with columns to be cleaned
    hints : bool, optional
        Print out hints on column name duplication and colum name length, by default \
        True

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with cleaned column names

    """
    _validate_input_bool(hints, "hints")

    # Handle CamelCase
    for i, col in enumerate(data.columns):
        matches = re.findall(re.compile("[a-z][A-Z]"), col)
        column = col
        for match in matches:
            column = column.replace(match, f"{match[0]}_{match[1]}")
            data = data.rename(columns={data.columns[i]: column})

    data.columns = (
        data.columns.str.replace("\n", "_", regex=False)
        .str.replace("(", "_", regex=False)
        .str.replace(")", "_", regex=False)
        .str.replace("'", "_", regex=False)
        .str.replace('"', "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"[!?:;/]", "_", regex=True)
        .str.replace("+", "_plus_", regex=False)
        .str.replace("*", "_times_", regex=False)
        .str.replace("<", "_smaller", regex=False)
        .str.replace(">", "_larger_", regex=False)
        .str.replace("=", "_equal_", regex=False)
        .str.replace("ä", "ae", regex=False)
        .str.replace("ö", "oe", regex=False)
        .str.replace("ü", "ue", regex=False)
        .str.replace("ß", "ss", regex=False)
        .str.replace("%", "_percent_", regex=False)
        .str.replace("$", "_dollar_", regex=False)
        .str.replace("€", "_euro_", regex=False)
        .str.replace("@", "_at_", regex=False)
        .str.replace("#", "_hash_", regex=False)
        .str.replace("&", "_and_", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )

    if dupl_idx := [i for i, x in enumerate(data.columns.duplicated()) if x]:
        dupl_before = data.columns[dupl_idx].tolist()
        data.columns = [
            col if col not in data.columns[:i] else f"{col}_{i!s}"
            for i, col in enumerate(data.columns)
        ]
        print_statement = ""
        if hints:
            # add string to print statement
            print_statement = (
                f"Duplicate column names detected! Columns with index {dupl_idx} and "
                f"names {dupl_before} have been renamed to "
                f"{data.columns[dupl_idx].tolist()}.",
            )

            if long_col_names := [
                x
                for x in data.columns
                if len(x) > 25  # noqa: PLR2004
            ]:
                print_statement += (
                    "Long column names detected (>25 characters). Consider renaming "
                    f"the following columns {long_col_names}.",
                )

            print(print_statement)

    return data


def convert_datatypes(
    data: pd.DataFrame,
    category: bool = True,
    cat_threshold: float = 0.05,
    cat_exclude: list[str | int] | None = None,
) -> pd.DataFrame:
    """Convert columns to best possible dtypes using dtypes supporting pd.NA.

    Temporarily not converting to integers due to an issue in pandas. This is expected \
        to be fixed in pandas 1.1. See https://github.com/pandas-dev/pandas/issues/33803

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    category : bool, optional
        Change dtypes of columns with dtype "object" to "category". Set threshold \
        using cat_threshold or exclude columns using cat_exclude, by default True
    cat_threshold : float, optional
        Ratio of unique values below which categories are inferred and column dtype is \
        changed to categorical, by default 0.05
    cat_exclude : Optional[list[str | int]], optional
        List of columns to exclude from categorical conversion, by default None

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with converted Datatypes

    """
    # Validate Inputs
    _validate_input_bool(category, "Category")
    _validate_input_range(cat_threshold, "cat_threshold", 0, 1)

    cat_exclude = [] if cat_exclude is None else cat_exclude.copy()

    data = pd.DataFrame(data).copy()
    for col in data.columns:
        unique_vals_ratio = data[col].nunique(dropna=False) / data.shape[0]
        if (
            category
            and unique_vals_ratio < cat_threshold
            and col not in cat_exclude
            and data[col].dtype == "object"
        ):
            data[col] = data[col].astype("category")

        data[col] = data[col].convert_dtypes(
            convert_integer=False,
            convert_floating=False,
        )

    data = _optimize_ints(data)
    return _optimize_floats(data)


def drop_missing(
    data: pd.DataFrame,
    drop_threshold_cols: float = 1,
    drop_threshold_rows: float = 1,
    col_exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Drop completely empty columns and rows by default.

    Optionally provides flexibility to loosen restrictions to drop additional \
    non-empty columns and rows based on the fraction of NA-values.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    drop_threshold_cols : float, optional
        Drop columns with NA-ratio equal to or above the specified threshold, by \
        default 1
    drop_threshold_rows : float, optional
        Drop rows with NA-ratio equal to or above the specified threshold, by default 1
    col_exclude : Optional[list[str]], optional
        Specify a list of columns to exclude from dropping. The excluded columns do \
        not affect the drop thresholds, by default None

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame without any empty columns or rows

    Notes
    -----
    Columns are dropped first

    """
    # Validate Inputs
    _validate_input_range(drop_threshold_cols, "drop_threshold_cols", 0, 1)
    _validate_input_range(drop_threshold_rows, "drop_threshold_rows", 0, 1)

    col_exclude = [] if col_exclude is None else col_exclude.copy()
    data_exclude = data[col_exclude]

    data = pd.DataFrame(data).copy()

    data_dropped = data.drop(columns=col_exclude, errors="ignore")
    data_dropped = data_dropped.drop(
        columns=data_dropped.loc[
            :,
            _missing_vals(data)["mv_cols_ratio"] > drop_threshold_cols,
        ].columns,
    ).dropna(axis=1, how="all")

    data = pd.concat([data_dropped, data_exclude], axis=1)

    return data.drop(
        index=data.loc[
            _missing_vals(data)["mv_rows_ratio"] > drop_threshold_rows,
            :,
        ].index,
    ).dropna(axis=0, how="all")


def data_cleaning(
    data: pd.DataFrame,
    drop_threshold_cols: float = 0.9,
    drop_threshold_rows: float = 0.9,
    drop_duplicates: bool = True,
    convert_dtypes: bool = True,
    col_exclude: list[str] | None = None,
    category: bool = True,
    cat_threshold: float = 0.03,
    cat_exclude: list[str | int] | None = None,
    clean_col_names: bool = True,
    show: Literal["all", "changes"] | None = "changes",
) -> pd.DataFrame:
    """Perform initial data cleaning tasks on a dataset.

    For example dropping single valued and empty rows, empty columns as well as \
    optimizing the datatypes.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    drop_threshold_cols : float, optional
        Drop columns with NA-ratio equal to or above the specified threshold, by \
        default 0.9
    drop_threshold_rows : float, optional
        Drop rows with NA-ratio equal to or above the specified threshold, by \
        default 0.9
    drop_duplicates : bool, optional
        Drop duplicate rows, keeping the first occurence. This step comes after the \
        dropping of missing values, by default True
    convert_dtypes : bool, optional
        Convert dtypes using pd.convert_dtypes(), by default True
    col_exclude : Optional[list[str]], optional
        Specify a list of columns to exclude from dropping, by default None
    category : bool, optional
        Enable changing dtypes of "object" columns to "category". Set threshold using \
        cat_threshold. Requires convert_dtypes=True, by default True
    cat_threshold : float, optional
        Ratio of unique values below which categories are inferred and column dtype is \
        changed to categorical, by default 0.03
    cat_exclude : Optional[list[str]], optional
        List of columns to exclude from categorical conversion, by default None
    clean_col_names: bool, optional
        Cleans the column names and provides hints on duplicate and long names, by \
        default True
    show : Optional[Literal["all", "changes"]], optional
        {"all", "changes", None}, by default "changes"
        Specify verbosity of the output:

            * "all": Print information about the data before and after cleaning as \
            well as information about  changes and memory usage (deep). Please be \
            aware, that this can slow down the function by quite a bit.
            * "changes": Print out differences in the data before and after cleaning.
            * None: No information about the data and the data cleaning is printed.

    Returns
    -------
    pd.DataFrame
        Cleaned Pandas DataFrame

    See Also
    --------
    convert_datatypes: Convert columns to best possible dtypes.
    drop_missing : Flexibly drop columns and rows.
    _memory_usage: Gives the total memory usage in megabytes.
    _missing_vals: Metrics about missing values in the dataset.

    Notes
    -----
    The category dtype is not grouped in the summary, unless it contains exactly the \
    same categories.

    """
    if col_exclude is None:
        col_exclude = []

    # Validate Inputs
    _validate_input_range(drop_threshold_cols, "drop_threshold_cols", 0, 1)
    _validate_input_range(drop_threshold_rows, "drop_threshold_rows", 0, 1)
    _validate_input_bool(drop_duplicates, "drop_duplicates")
    _validate_input_bool(convert_dtypes, "convert_datatypes")
    _validate_input_bool(category, "category")
    _validate_input_range(cat_threshold, "cat_threshold", 0, 1)

    data = pd.DataFrame(data).copy()
    data_cleaned = drop_missing(
        data,
        drop_threshold_cols,
        drop_threshold_rows,
        col_exclude=col_exclude,
    )

    if clean_col_names:
        data_cleaned = clean_column_names(data_cleaned)

    single_val_cols = data_cleaned.columns[
        data_cleaned.nunique(dropna=False) == 1  # noqa: PD101
    ].tolist()
    single_val_cols = [col for col in single_val_cols if col not in col_exclude]
    data_cleaned = data_cleaned.drop(columns=single_val_cols)

    dupl_rows = None

    if drop_duplicates:
        data_cleaned, dupl_rows = _drop_duplicates(data_cleaned)
    if convert_dtypes:
        data_cleaned = convert_datatypes(
            data_cleaned,
            category=category,
            cat_threshold=cat_threshold,
            cat_exclude=cat_exclude,
        )

    _diff_report(
        data,
        data_cleaned,
        dupl_rows=dupl_rows,
        single_val_cols=single_val_cols,
        show=show,
    )

    return data_cleaned


def mv_col_handling(
    data: pd.DataFrame,
    target: str | pd.Series | list[str] | None = None,
    mv_threshold: float = 0.1,
    corr_thresh_features: float = 0.5,
    corr_thresh_target: float = 0.3,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str], list[str]]:
    """Convert columns with a high ratio of missing values into binary features.

    Eventually drops them based on their correlation with other features and the \
    target variable.

    This function follows a three step process:
    - 1) Identify features with a high ratio of missing values (above 'mv_threshold').
    - 2) Identify high correlations of these features among themselves and with \
        other features in the dataset (above 'corr_thresh_features').
    - 3) Features with high ratio of missing values and high correlation among each \
        other are dropped unless they correlate reasonably well with the target \
        variable (above 'corr_thresh_target').

    Note: If no target is provided, the process exits after step two and drops columns \
    identified up to this point.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    target : Optional[str | pd.Series | list]], optional
        Specify target for correlation. I.e. label column to generate only the \
        correlations between each feature and the label, by default None
    mv_threshold : float, optional
        Value between 0 <= threshold <= 1. Features with a missing-value-ratio larger \
        than mv_threshold are candidates for dropping and undergo further analysis, by \
        default 0.1
    corr_thresh_features : float, optional
        Value between 0 <= threshold <= 1. Maximum correlation a previously identified \
        features (with a high mv-ratio) is allowed to have with another feature. If \
        this threshold is overstepped, the feature undergoes further analysis, by \
        default 0.5
    corr_thresh_target : float, optional
        Value between 0 <= threshold <= 1. Minimum required correlation of a remaining \
        feature (i.e. feature with a high mv-ratio and high correlation to another \
        existing feature) with the target. If this threshold is not met the feature is \
        ultimately dropped, by default 0.3
    return_details : bool, optional
        Provdies flexibility to return intermediary results, by default False

    Returns
    -------
    pd.DataFrame
        Updated Pandas DataFrame

    optional:
    cols_mv: Columns with missing values included in the analysis
    drop_cols: List of dropped columns

    """
    # Validate Inputs
    _validate_input_range(mv_threshold, "mv_threshold", 0, 1)
    _validate_input_range(corr_thresh_features, "corr_thresh_features", 0, 1)
    _validate_input_range(corr_thresh_target, "corr_thresh_target", 0, 1)

    data = pd.DataFrame(data).copy()
    data_local = data.copy()
    mv_ratios = _missing_vals(data_local)["mv_cols_ratio"]
    cols_mv = mv_ratios[mv_ratios > mv_threshold].index.tolist()
    data_local[cols_mv] = data_local[cols_mv].applymap(lambda x: x if pd.isna(x) else 1).fillna(0)

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
        corrs = corr_mat(data_local, target=target, colored=False).loc[high_corr_features]
        drop_cols = corrs.loc[abs(corrs.iloc[:, 0]) < corr_thresh_target].index.tolist()
        data = data.drop(columns=drop_cols)

    return (data, cols_mv, drop_cols) if return_details else data


def pool_duplicate_subsets(
    data: pd.DataFrame,
    col_dupl_thresh: float = 0.2,
    subset_thresh: float = 0.2,
    min_col_pool: int = 3,
    exclude: list[str] | None = None,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Check for duplicates in subsets of columns and pools them.

    This can reduce the number of columns in the data without loosing much \
    information. Suitable columns are combined to subsets and tested for duplicates. \
    In case sufficient duplicates can be found, the respective columns are aggregated \
    into a "pooled_var" column. Identical numbers in the "pooled_var" column indicate \
    identical information in the respective rows.

    Note:  It is advised to exclude features that provide sufficient informational \
    content by themselves as well as the target column by using the "exclude" \
    setting.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    col_dupl_thresh : float, optional
        Columns with a ratio of duplicates higher than "col_dupl_thresh" are \
        considered in the further analysis. Columns with a lower ratio are not \
        considered for pooling, by default 0.2
    subset_thresh : float, optional
        The first subset with a duplicate threshold higher than "subset_thresh" is \
        chosen and aggregated. If no subset reaches the threshold, the algorithm \
        continues with continuously smaller subsets until "min_col_pool" is reached, \
        by default 0.2
    min_col_pool : int, optional
        Minimum number of columns to pool. The algorithm attempts to combine as many \
        columns as possible to suitable subsets and stops when "min_col_pool" is \
        reached, by default 3
    exclude : Optional[list[str]], optional
        List of column names to be excluded from the analysis. These columns are \
        passed through without modification, by default None
    return_details : bool, optional
        Provdies flexibility to return intermediary results, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with low cardinality columns pooled

    optional:
    subset_cols: List of columns used as subset

    """
    # Input validation
    _validate_input_range(col_dupl_thresh, "col_dupl_thresh", 0, 1)
    _validate_input_range(subset_thresh, "subset_thresh", 0, 1)
    _validate_input_range(min_col_pool, "min_col_pool", 0, data.shape[1])

    excluded_cols = []
    if exclude is not None:
        excluded_cols = data[exclude]
        data = data.drop(columns=exclude)

    subset_cols = []
    for i in range(data.shape[1] + 1 - min_col_pool):
        # Consider only columns with lots of duplicates
        check = [
            col for col in data.columns if data.duplicated(subset=col).mean() > col_dupl_thresh
        ]

        # Identify all possible combinations for the current interation
        if check:
            combinations = itertools.combinations(check, len(check) - i)
        else:
            continue

        # Check subsets for all possible combinations
        ratios = [*(data.duplicated(subset=list(comb)).mean() for comb in combinations)]

        max_idx = np.argmax(ratios)

        if max(ratios) > subset_thresh:
            # Get the best possible iterator and process the data
            best_subset = itertools.islice(
                itertools.combinations(check, len(check) - i),
                max_idx,
                max_idx + 1,
            )

            best_subset = data[list(next(iter(best_subset)))]
            subset_cols = best_subset.columns.tolist()

            unique_subset = (
                best_subset.drop_duplicates().reset_index().rename(columns={"index": "pooled_vars"})
            )
            data = data.merge(unique_subset, how="left", on=subset_cols).drop(
                columns=subset_cols,
            )
            data.index = pd.RangeIndex(len(data))
            break

    data = pd.concat([data, pd.DataFrame(excluded_cols)], axis=1)

    return (data, subset_cols) if return_details else data
