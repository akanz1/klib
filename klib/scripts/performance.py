"""
Measuring the performance of key functionality.

:author: Andreas Kanz
"""

import functools
import pandas as pd
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt

import klib

# Paths
base_path = Path(__file__).resolve().parents[2]
print(base_path)
data_path = base_path / "examples"
export_path = base_path / "klib/scripts/"

# Data Import
filepath = data_path / "NFL_DATASET.csv"
data = pd.read_csv(filepath)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        time_start = perf_counter()
        func(*args, **kwargs)
        duration = perf_counter() - time_start
        return duration

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
    df_times["data_cleaning"] = pd.Series([time_data_cleaning() for _ in range(8)])
    df_times["missingval_plot"] = pd.Series([time_missingval_plot() for _ in range(8)])
    df_times["dist_plot"] = pd.Series([time_dist_plot() for _ in range(3)])
    print("----cat_plot----")
    df_times["cat_plot"] = pd.Series([time_cat_plot() for _ in range(8)])
    df_times = df_times.fillna(df_times.mean())
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13, 6))
    reference_values = [4.3, 6.4, 6, 5.6]

    for i, (col, ref) in enumerate(zip(df_times.columns, reference_values)):
        ax[i].boxplot(df_times[col])
        ax[i].set_title(" ".join(col.split("_")).title())
        ax[i].axhline(ref)
    fig.suptitle("Performance", fontsize=14)
    fig.savefig("boxplots.png")


if __name__ == "__main__":
    main()
