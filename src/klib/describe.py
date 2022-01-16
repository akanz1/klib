"""
Functions for descriptive analytics.

:author: Andreas Kanz

"""

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from klib.utils import (
    _corr_selector,
    _missing_vals,
    _validate_input_bool,
    _validate_input_int,
    _validate_input_range,
    _validate_input_smaller,
    _validate_input_sum_larger,
)

__all__ = ["cat_plot", "corr_mat", "corr_plot", "dist_plot", "missingval_plot"]


# Functions

# Categorical Plot
def cat_plot(
    data: pd.DataFrame,
    figsize: Tuple = (18, 18),
    top: int = 3,
    bottom: int = 3,
    bar_color_top: str = "#5ab4ac",
    bar_color_bottom: str = "#d8b365",
):
    """Two-dimensional visualization of the number and frequency of categorical \
        features.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
        is provided, the index/column information is used to label the plots
    figsize : Tuple, optional
        Use to control the figure size, by default (18, 18)
    top : int, optional
        Show the "top" most frequent values in a column, by default 3
    bottom : int, optional
        Show the "bottom" most frequent values in a column, by default 3
    bar_color_top : str, optional
        Use to control the color of the bars indicating the most common values, by \
        default "#5ab4ac"
    bar_color_bottom : str, optional
        Use to control the color of the bars indicating the least common values, by \
        default "#d8b365"

    Returns
    -------
    Gridspec
        gs: Figure with array of Axes objects
    """
    # Validate Inputs
    _validate_input_int(top, "top")
    _validate_input_int(bottom, "bottom")
    _validate_input_range(top, "top", 0, data.shape[1])
    _validate_input_range(bottom, "bottom", 0, data.shape[1])
    _validate_input_sum_larger(1, "top and bottom", top, bottom)

    data = pd.DataFrame(data).copy()
    cols = data.select_dtypes(exclude=["number"]).columns.tolist()
    data = data[cols]

    if len(cols) == 0:
        print("No columns with categorical data were detected.")
        return None

    for col in data.columns:
        if data[col].dtype.name in ("category", "string"):
            data[col] = data[col].astype("object")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=6, ncols=len(cols), wspace=0.21)

    for count, col in enumerate(cols):
        n_unique = data[col].nunique(dropna=True)
        value_counts = data[col].value_counts()
        lim_top, lim_bot = top, bottom

        if n_unique < top + bottom:
            lim_top = int(n_unique // 2)
            lim_bot = int(n_unique // 2) + 1

        if n_unique <= 2:
            lim_top = lim_bot = int(n_unique // 2)

        value_counts_top = value_counts[:lim_top]
        value_counts_idx_top = value_counts_top.index.tolist()
        value_counts_bot = value_counts[-lim_bot:]
        value_counts_idx_bot = value_counts_bot.index.tolist()

        if top == 0:
            value_counts_top = value_counts_idx_top = []

        if bottom == 0:
            value_counts_bot = value_counts_idx_bot = []

        data.loc[data[col].isin(value_counts_idx_top), col] = 10
        data.loc[data[col].isin(value_counts_idx_bot), col] = 0
        data.loc[((data[col] != 10) & (data[col] != 0)), col] = 5
        data[col] = data[col].rolling(2, min_periods=1).mean()

        value_counts_idx_top = [elem[:20] for elem in value_counts_idx_top]
        value_counts_idx_bot = [elem[:20] for elem in value_counts_idx_bot]
        sum_top = sum(value_counts_top)
        sum_bot = sum(value_counts_bot)

        # Barcharts
        ax_top = fig.add_subplot(gs[:1, count : count + 1])
        ax_top.bar(
            value_counts_idx_top, value_counts_top, color=bar_color_top, width=0.85
        )
        ax_top.bar(
            value_counts_idx_bot, value_counts_bot, color=bar_color_bottom, width=0.85
        )
        ax_top.set(frame_on=False)
        ax_top.tick_params(axis="x", labelrotation=90)

        # Summary stats
        ax_bottom = fig.add_subplot(gs[1:2, count : count + 1])
        plt.subplots_adjust(hspace=0.075)
        ax_bottom.get_yaxis().set_visible(False)
        ax_bottom.get_xaxis().set_visible(False)
        ax_bottom.set(frame_on=False)
        ax_bottom.text(
            0,
            0,
            f"Unique values: {n_unique}\n\n"
            f"Top {lim_top} vals: {sum_top} ({sum_top/data.shape[0]*100:.1f}%)\n"
            f"Bot {lim_bot} vals: {sum_bot} ({sum_bot/data.shape[0]*100:.1f}%)",
            transform=ax_bottom.transAxes,
            color="#111111",
            fontsize=11,
        )

    # Heatmap
    color_bot_rgb = to_rgb(bar_color_bottom)
    color_white = to_rgb("#FFFFFF")
    color_top_rgb = to_rgb(bar_color_top)
    cat_plot_cmap = LinearSegmentedColormap.from_list(
        "cat_plot_cmap", [color_bot_rgb, color_white, color_top_rgb], N=200
    )
    ax_hm = fig.add_subplot(gs[2:, :])
    sns.heatmap(data, cmap=cat_plot_cmap, cbar=False, vmin=0, vmax=10, ax=ax_hm)
    ax_hm.set_yticks(np.round(ax_hm.get_yticks()[0::5], -1))
    ax_hm.set_yticklabels(ax_hm.get_yticks())
    ax_hm.set_xticklabels(
        ax_hm.get_xticklabels(),
        horizontalalignment="center",
        fontweight="light",
        fontsize="medium",
    )
    ax_hm.tick_params(length=1, colors="#111111")
    gs.figure.suptitle(
        "Categorical data plot", x=0.5, y=0.91, fontsize=18, color="#111111"
    )

    return gs


# Correlation Matrix
def corr_mat(
    data: pd.DataFrame,
    split: Optional[
        str
    ] = None,  # Optional[Literal['pos', 'neg', 'high', 'low']] = None,
    threshold: float = 0,
    target: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, str]] = None,
    method: str = "pearson",  # Literal['pearson', 'spearman', 'kendall'] = "pearson",
    colored: bool = True,
) -> Union[pd.DataFrame, Any]:
    """Return a color-encoded correlation matrix.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
        is provided, the index/column information is used to label the plots
    split : Optional[str], optional
        Type of split to be performed, by default None
        {None, "pos", "neg", "high", "low"}
    threshold : float, optional
        Value between 0 and 1 to set the correlation threshold, by default 0 unless \
        split = "high" or split = "low", in which case default is 0.3
    target : Optional[Union[pd.DataFrame, str]], optional
        Specify target for correlation. E.g. label column to generate only the \
        correlations between each feature and the label, by default None
    method : str, optional
        method: {"pearson", "spearman", "kendall"}, by default "pearson"
        * pearson: measures linear relationships and requires normally distributed \
            and homoscedastic data.
        * spearman: ranked/ordinal correlation, measures monotonic relationships.
        * kendall: ranked/ordinal correlation, measures monotonic relationships. \
            Computationally more expensive but more robust in smaller dataets than \
            "spearman"
    colored : bool, optional
        If True the negative values in the correlation matrix are colored in red, by \
        default True

    Returns
    -------
    Union[pd.DataFrame, pd.Styler]
        If colored = True - corr: Pandas Styler object
        If colored = False - corr: Pandas DataFrame
    """
    # Validate Inputs
    _validate_input_range(threshold, "threshold", -1, 1)
    _validate_input_bool(colored, "colored")

    def color_negative_red(val):
        color = "#FF3344" if val < 0 else None
        return f"color: {color}"

    data = pd.DataFrame(data)

    if isinstance(target, (str, list, pd.Series, np.ndarray)):
        target_data = []
        if isinstance(target, str):
            target_data = data[target]
            data = data.drop(target, axis=1)

        elif isinstance(target, (list, pd.Series, np.ndarray)):
            target_data = pd.Series(target)
            target = target_data.name

        corr = pd.DataFrame(data.corrwith(target_data, method=method))
        corr = corr.sort_values(corr.columns[0], ascending=False)
        corr.columns = [target]

    else:
        corr = data.corr(method=method)

    corr = _corr_selector(corr, split=split, threshold=threshold)

    if colored:
        return corr.style.applymap(color_negative_red).format("{:.2f}", na_rep="-")
    return corr


# Correlation matrix / heatmap
def corr_plot(
    data: pd.DataFrame,
    split: Optional[str] = None,
    threshold: float = 0,
    target: Optional[Union[pd.Series, str]] = None,
    method: str = "pearson",
    cmap: str = "BrBG",
    figsize: Tuple = (12, 10),
    annot: bool = True,
    dev: bool = False,
    **kwargs,
):
    """Two-dimensional visualization of the correlation between feature-columns \
        excluding NA values.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
        is provided, the index/column information is used to label the plots
    split : Optional[str], optional
        Type of split to be performed {None, "pos", "neg", "high", "low"}, by default \
        None
            * None: visualize all correlations between the feature-columns
            * pos: visualize all positive correlations between the feature-columns \
                above the threshold
            * neg: visualize all negative correlations between the feature-columns \
                below the threshold
            * high: visualize all correlations between the feature-columns for \
                which abs (corr) > threshold is True
            * low: visualize all correlations between the feature-columns for which \
                abs(corr) < threshold is True

    threshold : float, optional
        Value between 0 and 1 to set the correlation threshold, by default 0 unless \
            split = "high" or split = "low", in which case default is 0.3
    target : Optional[Union[pd.Series, str]], optional
        Specify target for correlation. E.g. label column to generate only the \
        correlations between each feature and the label, by default None
    method : str, optional
        method: {"pearson", "spearman", "kendall"}, by default "pearson"
            * pearson: measures linear relationships and requires normally \
                distributed and homoscedastic data.
            * spearman: ranked/ordinal correlation, measures monotonic relationships.
            * kendall: ranked/ordinal correlation, measures monotonic relationships. \
                Computationally more expensive but more robust in smaller dataets \
                than "spearman".

    cmap : str, optional
        The mapping from data values to color space, matplotlib colormap name or \
        object, or list of colors, by default "BrBG"
    figsize : Tuple, optional
        Use to control the figure size, by default (12, 10)
    annot : bool, optional
        Use to show or hide annotations, by default True
    dev : bool, optional
        Display figure settings in the plot by setting dev = True. If False, the \
        settings are not displayed, by default False

    Keyword Arguments : optional
        Additional elements to control the visualization of the plot, e.g.:

            * mask: bool, default True
                If set to False the entire correlation matrix, including the upper \
                triangle is shown. Set dev = False in this case to avoid overlap.
            * vmax: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 or vmin <= vmax <= 1, limits the range of the cbar.
            * vmin: float, default is calculated from the given correlation \
                coefficients.
                Value between -1 <= vmin <= 1 or vmax, limits the range of the cbar.
            * linewidths: float, default 0.5
                Controls the line-width inbetween the squares.
            * annot_kws: dict, default {"size" : 10}
                Controls the font size of the annotations. Only available when \
                annot = True.
            * cbar_kws: dict, default {"shrink": .95, "aspect": 30}
                Controls the size of the colorbar.
            * Many more kwargs are available, i.e. "alpha" to control blending, or \
                options to adjust labels, ticks ...

        Kwargs can be supplied through a dictionary of key-value pairs (see above).

    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    """
    # Validate Inputs
    _validate_input_range(threshold, "threshold", -1, 1)
    _validate_input_bool(annot, "annot")
    _validate_input_bool(dev, "dev")

    data = pd.DataFrame(data)

    corr = corr_mat(
        data,
        split=split,
        threshold=threshold,
        target=target,
        method=method,
        colored=False,
    )

    mask = np.zeros_like(corr, dtype=np.bool)

    if target is None:
        mask = np.triu(np.ones_like(corr, dtype=np.bool))

    vmax = np.round(np.nanmax(corr.where(~mask)) - 0.05, 2)
    vmin = np.round(np.nanmin(corr.where(~mask)) + 0.05, 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Specify kwargs for the heatmap
    kwargs = {
        "mask": mask,
        "cmap": cmap,
        "annot": annot,
        "vmax": vmax,
        "vmin": vmin,
        "linewidths": 0.5,
        "annot_kws": {"size": 10},
        "cbar_kws": {"shrink": 0.95, "aspect": 30},
        **kwargs,
    }

    # Draw heatmap with mask and default settings
    sns.heatmap(corr, center=0, fmt=".2f", **kwargs)

    ax.set_title(f"Feature-correlation ({method})", fontdict={"fontsize": 18})

    # Settings
    if dev:
        fig.suptitle(
            f"\
            Settings (dev-mode): \n\
            - split-mode: {split} \n\
            - threshold: {threshold} \n\
            - method: {method} \n\
            - annotations: {annot} \n\
            - cbar: \n\
                - vmax: {vmax} \n\
                - vmin: {vmin} \n\
            - linewidths: {kwargs['linewidths']} \n\
            - annot_kws: {kwargs['annot_kws']} \n\
            - cbar_kws: {kwargs['cbar_kws']}",
            fontsize=12,
            color="gray",
            x=0.35,
            y=0.85,
            ha="left",
        )

    return ax


# Distribution plot
def dist_plot(
    data: pd.DataFrame,
    mean_color: str = "orange",
    size: int = 2.5,
    fill_range: Tuple = (0.025, 0.975),
    showall: bool = False,
    kde_kws: Dict[str, Any] = None,
    rug_kws: Dict[str, Any] = None,
    fill_kws: Dict[str, Any] = None,
    font_kws: Dict[str, Any] = None,
):
    """Two-dimensional visualization of the distribution of non binary numerical \
        features.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
        is provided, the index/column information is used to label the plots
    mean_color : str, optional
        Color of the vertical line indicating the mean of the data, by default "orange"
    size : int, optional
        Controls the plot size, by default 2.5
    fill_range : Tuple, optional
        Set the quantiles for shading. Default spans 95% of the data, which is about \
        two std. deviations above and below the mean, by default (0.025, 0.975)
    showall : bool, optional
        Set to True to remove the output limit of 20 plots, by default False
    kde_kws : Dict[str, Any], optional
        Keyword arguments for kdeplot(), by default {"color": "k", "alpha": 0.75, \
        "linewidth": 1.5, "bw_adjust": 0.8}
    rug_kws : Dict[str, Any], optional
        Keyword arguments for rugplot(), by default {"color": "#ff3333", \
        "alpha": 0.15, "lw": 3, "height": 0.075}
    fill_kws : Dict[str, Any], optional
        Keyword arguments to control the fill, by default {"color": "#80d4ff", \
        "alpha": 0.2}
    font_kws : Dict[str, Any], optional
        Keyword arguments to control the font, by default {"color":  "#111111", \
        "weight": "normal", "size": 11}

    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    """
    # Validate Inputs
    _validate_input_range(fill_range[0], "fill_range_lower", 0, 1)
    _validate_input_range(fill_range[1], "fill_range_upper", 0, 1)
    _validate_input_smaller(fill_range[0], fill_range[1], "fill_range")
    _validate_input_bool(showall, "showall")

    # Handle dictionary defaults
    kde_kws = (
        {"alpha": 0.75, "linewidth": 1.5, "bw_adjust": 0.8}
        if kde_kws is None
        else kde_kws.copy()
    )
    rug_kws = (
        {"color": "#ff3333", "alpha": 0.15, "lw": 3, "height": 0.075}
        if rug_kws is None
        else rug_kws.copy()
    )
    fill_kws = (
        {"color": "#80d4ff", "alpha": 0.2} if fill_kws is None else fill_kws.copy()
    )
    font_kws = (
        {"color": "#111111", "weight": "normal", "size": 11}
        if font_kws is None
        else font_kws.copy()
    )

    data = pd.DataFrame(data.copy()).dropna(axis=1, how="all")
    df = data.copy()
    data = data.loc[:, data.nunique() > 2]
    if data.shape[0] > 10000:
        data = data.sample(n=10000, random_state=408)
        print(
            "Large dataset detected, using 10000 random samples for the plots. Summary"
            " statistics are still based on the entire dataset."
        )
    cols = list(data.select_dtypes(include=["number"]).columns)
    data = data[cols]

    if not cols:
        print("No columns with numeric data were detected.")
        return None

    if len(cols) >= 20 and not showall:
        print(
            "Note: The number of non binary numerical features is very large "
            f"({len(cols)}), please consider splitting the data. Showing plots for "
            "the first 20 numerical features. Override this by setting showall=True."
        )
        cols = cols[:20]

    g = None
    for col in cols:
        col_data = data[col].dropna(axis=0)
        col_df = df[col].dropna(axis=0)

        g = sns.displot(
            col_data,
            kind="kde",
            rug=True,
            height=size,
            aspect=5,
            legend=False,
            rug_kws=rug_kws,
            **kde_kws,
        )

        # Vertical lines and fill
        x, y = g.axes[0, 0].lines[0].get_xydata().T
        g.axes[0, 0].fill_between(
            x,
            y,
            where=(
                (x >= np.quantile(col_df, fill_range[0]))
                & (x <= np.quantile(col_df, fill_range[1]))
            ),
            label=f"{fill_range[0]*100:.1f}% - {fill_range[1]*100:.1f}%",
            **fill_kws,
        )

        mean = np.mean(col_df)
        std = scipy.stats.tstd(col_df)
        g.axes[0, 0].vlines(
            x=mean,
            ymin=0,
            ymax=np.interp(mean, x, y),
            ls="dotted",
            color=mean_color,
            lw=2,
            label="mean",
        )
        g.axes[0, 0].vlines(
            x=np.median(col_df),
            ymin=0,
            ymax=np.interp(np.median(col_df), x, y),
            ls=":",
            color=".3",
            label="median",
        )
        g.axes[0, 0].vlines(
            x=[mean - std, mean + std],
            ymin=0,
            ymax=[np.interp(mean - std, x, y), np.interp(mean + std, x, y)],
            ls=":",
            color=".5",
            label="\u03BC \u00B1 \u03C3",
        )

        g.axes[0, 0].set_ylim(0)
        g.axes[0, 0].set_xlim(
            g.axes[0, 0].get_xlim()[0] - g.axes[0, 0].get_xlim()[1] * 0.05,
            g.axes[0, 0].get_xlim()[1] * 1.03,
        )

        # Annotations and legend
        g.axes[0, 0].text(
            0.005,
            0.9,
            f"Mean: {mean:.2f}",
            fontdict=font_kws,
            transform=g.axes[0, 0].transAxes,
        )
        g.axes[0, 0].text(
            0.005,
            0.7,
            f"Std. dev: {std:.2f}",
            fontdict=font_kws,
            transform=g.axes[0, 0].transAxes,
        )
        g.axes[0, 0].text(
            0.005,
            0.5,
            f"Skew: {scipy.stats.skew(col_df):.2f}",
            fontdict=font_kws,
            transform=g.axes[0, 0].transAxes,
        )
        g.axes[0, 0].text(
            0.005,
            0.3,
            f"Kurtosis: {scipy.stats.kurtosis(col_df):.2f}",  # Excess Kurtosis
            fontdict=font_kws,
            transform=g.axes[0, 0].transAxes,
        )
        g.axes[0, 0].text(
            0.005,
            0.1,
            f"Count: {len(col_df)}",
            fontdict=font_kws,
            transform=g.axes[0, 0].transAxes,
        )
        g.axes[0, 0].legend(loc="upper right")

    return g.axes[0, 0]


# Missing value plot
def missingval_plot(
    data: pd.DataFrame,
    cmap: str = "PuBuGn",
    figsize: Tuple = (20, 20),
    sort: bool = False,
    spine_color: str = "#EEEEEE",
):
    """Two-dimensional visualization of the missing values in a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame \
        is provided, the index/column information is used to label the plots
    cmap : str, optional
        Any valid colormap can be used. E.g. "Greys", "RdPu". More information can be \
        found in the matplotlib documentation, by default "PuBuGn"
    figsize : Tuple, optional
        Use to control the figure size, by default (20, 20)
    sort : bool, optional
        Sort columns based on missing values in descending order and drop columns \
        without any missing values, by default False
    spine_color : str, optional
        Set to "None" to hide the spines on all plots or use any valid matplotlib \
        color argument, by default "#EEEEEE"

    Returns
    -------
    GridSpec
        gs: Figure with array of Axes objects
    """
    # Validate Inputs
    _validate_input_bool(sort, "sort")

    data = pd.DataFrame(data)

    if sort:
        mv_cols_sorted = data.isna().sum(axis=0).sort_values(ascending=False)
        final_cols = (
            mv_cols_sorted.drop(
                mv_cols_sorted[mv_cols_sorted.values == 0].keys().tolist()
            )
            .keys()
            .tolist()
        )
        data = data[final_cols]
        print("Displaying only columns with missing values.")

    # Identify missing values
    mv_total, mv_rows, mv_cols, _, mv_cols_ratio = _missing_vals(data).values()
    total_datapoints = data.shape[0] * data.shape[1]

    if mv_total == 0:
        print("No missing values found in the dataset.")
        return None

    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=6, ncols=6, left=0.1, wspace=0.05)
    ax1 = fig.add_subplot(gs[:1, :5])
    ax2 = fig.add_subplot(gs[1:, :5])
    ax3 = fig.add_subplot(gs[:1, 5:])
    ax4 = fig.add_subplot(gs[1:, 5:])

    # ax1 - Barplot
    colors = plt.get_cmap(cmap)(mv_cols / np.max(mv_cols))  # color bars by height
    ax1.bar(range(len(mv_cols)), np.round((mv_cols_ratio) * 100, 2), color=colors)
    ax1.get_xaxis().set_visible(False)
    ax1.set(frame_on=False, xlim=(-0.5, len(mv_cols) - 0.5))
    ax1.set_ylim(0, np.max(mv_cols_ratio) * 100)
    ax1.grid(linestyle=":", linewidth=1)
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
    ax1.tick_params(axis="y", colors="#111111", length=1)

    # annotate values on top of the bars
    for rect, label in zip(ax1.patches, mv_cols):
        height = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            height + max(np.log(1 + height / 6), 0.075),
            label,
            ha="center",
            va="bottom",
            rotation="90",
            alpha=0.5,
            fontsize="11",
        )

    ax1.set_frame_on(True)
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_color(spine_color)
    ax1.spines["top"].set_color(None)

    # ax2 - Heatmap
    sns.heatmap(data.isna(), cbar=False, cmap="binary", ax=ax2)
    ax2.set_yticks(np.round(ax2.get_yticks()[0::5], -1))
    ax2.set_yticklabels(ax2.get_yticks())
    ax2.set_xticklabels(
        ax2.get_xticklabels(),
        horizontalalignment="center",
        fontweight="light",
        fontsize="12",
    )
    ax2.tick_params(length=1, colors="#111111")
    for _, spine in ax2.spines.items():
        spine.set_visible(True)
        spine.set_color(spine_color)

    # ax3 - Summary
    fontax3 = {"color": "#111111", "weight": "normal", "size": 14}
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set(frame_on=False)

    ax3.text(
        0.025,
        0.875,
        f"Total: {np.round(total_datapoints/1000,1)}K",
        transform=ax3.transAxes,
        fontdict=fontax3,
    )
    ax3.text(
        0.025,
        0.675,
        f"Missing: {np.round(mv_total/1000,1)}K",
        transform=ax3.transAxes,
        fontdict=fontax3,
    )
    ax3.text(
        0.025,
        0.475,
        f"Relative: {np.round(mv_total/total_datapoints*100,1)}%",
        transform=ax3.transAxes,
        fontdict=fontax3,
    )
    ax3.text(
        0.025,
        0.275,
        f"Max-col: {np.round(mv_cols.max()/data.shape[0]*100)}%",
        transform=ax3.transAxes,
        fontdict=fontax3,
    )
    ax3.text(
        0.025,
        0.075,
        f"Max-row: {np.round(mv_rows.max()/data.shape[1]*100)}%",
        transform=ax3.transAxes,
        fontdict=fontax3,
    )

    # ax4 - Scatter plot
    ax4.get_yaxis().set_visible(False)
    for _, spine in ax4.spines.items():
        spine.set_color(spine_color)
    ax4.tick_params(axis="x", colors="#111111", length=1)

    ax4.scatter(
        mv_rows,
        range(len(mv_rows)),
        s=mv_rows,
        c=mv_rows,
        cmap=cmap,
        marker=".",
        vmin=1,
    )
    ax4.set_ylim((0, len(mv_rows))[::-1])  # limit and invert y-axis
    ax4.set_xlim(0, max(mv_rows) + 0.5)
    ax4.grid(linestyle=":", linewidth=1)

    gs.figure.suptitle(
        "Missing value plot", x=0.45, y=0.94, fontsize=18, color="#111111"
    )

    return gs
