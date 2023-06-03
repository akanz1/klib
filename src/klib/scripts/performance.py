"""Measuring the performance of key functionality.

:author: Andreas Kanz
"""
import functools
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

import klib

# Paths
base_path = Path(__file__).resolve().parents[3]
print(base_path)
data_path = base_path / "examples"

# Data Import
filepath = data_path / "NFL_DATASET.csv"
data = pd.read_csv(filepath)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        time_start = perf_counter()
        func(*args, **kwargs)
        return perf_counter() - time_start

    return wrapper


@timer
def time_data_cleaning():
    klib.data_cleaning(data, show=None)


@timer
def time_missingval_plot():
    klib.missingval_plot(data)


@timer
def time_dist_plot():
    klib.dist_plot(data.iloc[:, :5])


@timer
def time_cat_plot():
    klib.cat_plot(data)


def main():
    df_times = pd.DataFrame()
    df_times["data_cleaning"] = pd.Series([time_data_cleaning() for _ in range(12)])
    df_times["missingval_plot"] = pd.Series([time_missingval_plot() for _ in range(7)])
    df_times["dist_plot"] = pd.Series([time_dist_plot() for _ in range(7)])
    df_times["cat_plot"] = pd.Series([time_cat_plot() for _ in range(7)])
    df_times = df_times.fillna(df_times.mean())
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 7))
    reference_values = [5, 10, 10, 10]

    for i, (col, ref) in enumerate(
        zip(df_times.columns, reference_values, strict=True),
    ):
        ax[i].boxplot(df_times[col])
        ax[i].set_title(" ".join(col.split("_")).title())
        ax[i].axhline(ref)
    fig.suptitle("Performance", fontsize=16)
    fig.savefig("boxplots.png")


if __name__ == "__main__":
    main()
