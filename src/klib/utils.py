"""
Utilities and auxiliary functions.

:author: Andreas Kanz

"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _corr_selector(
    corr: Union[pd.Series, pd.DataFrame],
    split: Optional[
        str
    ] = None,  # Optional[Literal["pos", "neg", "high", "low"]] = None, Req:Python 3.8
    threshold: float = 0,
) -> Union[pd.Series, pd.DataFrame]:
    """Select the desired correlations using this utility function.

    Parameters
    ----------
    corr : Union[pd.Series, pd.DataFrame]
        pd.Series or pd.DataFrame of correlations
    split : Optional[str], optional
        Type of split performed, by default None
            * {None, "pos", "neg", "high", "low"}
    threshold : float, optional
        Value between 0 and 1 to set the correlation threshold, by default 0 unless \
        split = "high" or split = "low", in which case default is 0.3

    Returns
    -------
    pd.DataFrame
        List or matrix of (filtered) correlations
    """
    if split == "pos":
        corr = corr.where((corr >= threshold) & (corr > 0))
        print(
            'Displaying positive correlations. Specify a positive "threshold" to '
            "limit the results further."
        )
    elif split == "neg":
        corr = corr.where((corr <= threshold) & (corr < 0))
        print(
            'Displaying negative correlations. Specify a negative "threshold" to '
            "limit the results further."
        )
    elif split == "high":
        threshold = 0.3 if threshold <= 0 else threshold
        corr = corr.where(np.abs(corr) >= threshold)
        print(
            f"Displaying absolute correlations above the threshold ({threshold}). "
            'Specify a positive "threshold" to limit the results further.'
        )
    elif split == "low":
        threshold = 0.3 if threshold <= 0 else threshold
        corr = corr.where(np.abs(corr) <= threshold)
        print(
            f"Displaying absolute correlations below the threshold ({threshold}). "
            'Specify a positive "threshold" to limit the results further.'
        )

    return corr


def _diff_report(
    data: pd.DataFrame,
    data_cleaned: pd.DataFrame,
    dupl_rows: Optional[List[Union[str, int]]] = None,
    single_val_cols: Optional[List[str]] = None,
    show: Optional[str] = "changes",  # Optional[Literal["all", "changes"]] = "changes"
) -> None:
    """Provide information about changes between two datasets, such as dropped rows \
        and columns, memory usage and missing values.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. Input the initial \
        dataset here
    data_cleaned : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. Input the cleaned / \
        updated dataset here
    dupl_rows : Optional[List[Union[str, int]]], optional
        List of duplicate row indices, by default None
    single_val_cols : Optional[List[str]], optional
        List of single-valued column indices. I.e. columns where all cells contain \
        the same value. NaNs count as a separate value, by default None
    show : str, optional
        {"all", "changes", None}, by default "changes"
        Specify verbosity of the output:
            * "all": Print information about the data before and after cleaning as \
                well as information about changes and memory usage (deep). Please be \
                aware, that this can slow down the function by quite a bit.
            * "changes": Print out differences in the data before and after cleaning.
            * None: No information about the data and the data cleaning is printed.

    Returns
    -------
    None
        Print statement highlighting the datasets or changes between the two datasets.
    """
    if show not in ["changes", "all"]:
        return

    dupl_rows = [] if dupl_rows is None else dupl_rows.copy()
    single_val_cols = [] if single_val_cols is None else single_val_cols.copy()
    data_mem = _memory_usage(data, deep=False)
    data_cl_mem = _memory_usage(data_cleaned, deep=False)
    data_mv_tot = _missing_vals(data)["mv_total"]
    data_cl_mv_tot = _missing_vals(data_cleaned)["mv_total"]

    if show == "all":
        data_mem = _memory_usage(data, deep=True)
        data_cl_mem = _memory_usage(data_cleaned, deep=True)
        print("Before data cleaning:\n")
        print(f"dtypes:\n{data.dtypes.value_counts()}")
        print(f"\nNumber of rows: {str(data.shape[0]).rjust(8)}")
        print(f"Number of cols: {str(data.shape[1]).rjust(8)}")
        print(f"Missing values: {str(data_mv_tot).rjust(8)}")
        print(f"Memory usage: {str(data_mem).rjust(7)} MB")
        print("_______________________________________________________\n")
        print("After data cleaning:\n")
        print(f"dtypes:\n{data_cleaned.dtypes.value_counts()}")
        print(f"\nNumber of rows: {str(data_cleaned.shape[0]).rjust(8)}")
        print(f"Number of cols: {str(data_cleaned.shape[1]).rjust(8)}")
        print(f"Missing values: {str(data_cl_mv_tot).rjust(8)}")
        print(f"Memory usage: {str(data_cl_mem).rjust(7)} MB")
        print("_______________________________________________________\n")

    print(
        f"Shape of cleaned data: {data_cleaned.shape}"
        f"Remaining NAs: {data_cl_mv_tot}"
    )
    print("\nChanges:")
    print(f"Dropped rows: {data.shape[0]-data_cleaned.shape[0]}")
    print(f"     of which {len(dupl_rows)} duplicates. (Rows: {dupl_rows[:200]})")
    print(f"Dropped columns: {data.shape[1]-data_cleaned.shape[1]}")
    print(
        f"     of which {len(single_val_cols)} single valued."
        f"     Columns: {single_val_cols}"
    )
    print(f"Dropped missing values: {data_mv_tot-data_cl_mv_tot}")
    mem_change = data_mem - data_cl_mem
    mem_perc = round(100 * mem_change / data_mem, 2)
    print(f"Reduced memory by at least: {round(mem_change,3)} MB (-{mem_perc}%)\n")


def _drop_duplicates(data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
    """Provide information on and drops duplicate rows.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame

    Returns
    -------
    Tuple[pd.DataFrame, List]
        Deduplicated Pandas DataFrame and Index Object of rows dropped
    """
    data = pd.DataFrame(data).copy()
    dupl_rows = data[data.duplicated()].index.tolist()
    data = data.drop(dupl_rows, axis="index").reset_index(drop=True)

    return data, dupl_rows


def _memory_usage(data: pd.DataFrame, deep: bool = True) -> float:
    """Give the total memory usage in megabytes.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame
    deep : bool, optional
        Runs a deep analysis of the memory usage, by default True

    Returns
    -------
    float
        Memory usage in megabytes
    """
    data = pd.DataFrame(data).copy()
    return round(data.memory_usage(index=True, deep=deep).sum() / (1024 ** 2), 2)


def _missing_vals(data: pd.DataFrame) -> Dict[str, Any]:
    """Give metrics of missing values in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame

    Returns
    -------
    Dict[str, float]
        mv_total: float, number of missing values in the entire dataset
        mv_rows: float, number of missing values in each row
        mv_cols: float, number of missing values in each column
        mv_rows_ratio: float, ratio of missing values for each row
        mv_cols_ratio: float, ratio of missing values for each column
    """
    data = pd.DataFrame(data).copy()
    mv_rows = data.isna().sum(axis=1)
    mv_cols = data.isna().sum(axis=0)
    mv_total = data.isna().sum().sum()
    mv_rows_ratio = mv_rows / data.shape[1]
    mv_cols_ratio = mv_cols / data.shape[0]

    return {
        "mv_total": mv_total,
        "mv_rows": mv_rows,
        "mv_cols": mv_cols,
        "mv_rows_ratio": mv_rows_ratio,
        "mv_cols_ratio": mv_cols_ratio,
    }


def _validate_input_bool(value, desc):
    if not isinstance(value, bool):
        raise TypeError(
            f"Input value for '{desc}' is {type(value)} but should be a boolean."
        )


def _validate_input_int(value, desc):
    if not isinstance(value, int):
        raise TypeError(
            f"Input value for '{desc}' is {type(value)} but should be an integer."
        )


def _validate_input_range(value, desc, lower, upper):
    if value < lower or value > upper:
        raise ValueError(
            f"'{desc}' = {value} but should be {lower} <= '{desc}' <= {upper}."
        )


def _validate_input_smaller(value1, value2, desc):
    if value1 > value2:
        raise ValueError(
            f"The first input for '{desc}' should be smaller or equal to the second."
        )


def _validate_input_sum_smaller(limit, desc, *args):
    if sum(args) > limit:
        raise ValueError(
            f"The sum of input values for '{desc}' should be less or equal to {limit}."
        )


def _validate_input_sum_larger(limit, desc, *args):
    if sum(args) < limit:
        raise ValueError(
            f"The sum of input values for '{desc}' should be larger/equal to {limit}."
        )
