'''
Functions for descriptive analytics.

:author: Andreas Kanz

'''

# Imports
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from .utils import _corr_selector
from .utils import _missing_vals
from .utils import _validate_input_0_1
from .utils import _validate_input_bool


# Functions

# Correlation Matrix
def corr_mat(data, split=None, threshold=0, method='pearson'):
    '''
    Parameters
    ----------

    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame is provided, the index/column \
    information is used to label the plots.

    split: {None, 'pos', 'neg', 'high', 'low'}, default None
        Type of split to be performed.

    threshold: float, default 0
        Value between 0 <= threshold <= 1

    method: {'pearson', 'spearman', 'kendall'}, default 'pearson'
        * pearson: measures linear relationships and requires normally distributed and homoscedastic data.
        * spearman: ranked/ordinal correlation, measures monotonic relationships.
        * kendall: ranked/ordinal correlation, measures monotonic relationships. Computationally more expensive but
                    more robus in smaller dataets than 'spearman'.

    Returns
    -------
    returns a Pandas Styler object
    '''

    # Validate Inputs
    _validate_input_0_1(threshold, 'threshold')

    def color_negative_red(val):
        color = '#FF3344' if val < 0 else None
        return 'color: %s' % color

    data = pd.DataFrame(data)
    corr = data.corr(method=method)

    corr = _corr_selector(corr, split=split, threshold=threshold)

    return corr.style.applymap(color_negative_red).format("{:.2f}", na_rep='-')


# Correlation matrix / heatmap
def corr_plot(data, split=None, threshold=0, target=None, method='pearson', cmap='BrBG', figsize=(12, 10), annot=True,
              dev=False, **kwargs):
    '''
    Two-dimensional visualization of the correlation between feature-columns, excluding NA values.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame is provided, the index/column \
    information is used to label the plots.

    split: {None, 'pos', 'neg', 'high', 'low'}, default None
        Type of split to be performed.

        * None: visualize all correlations between the feature-columns.
        * pos: visualize all positive correlations between the feature-columns above the threshold.
        * neg: visualize all negative correlations between the feature-columns below the threshold.
        * high: visualize all correlations between the feature-columns for which abs(corr) > threshold is True.
        * low: visualize all correlations between the feature-columns for which abs(corr) < threshold is True.

    threshold: float, default 0
        Value between 0 <= threshold <= 1

    target: string, list, np.array or pd.Series, default None
        Specify target for correlation. E.g. label column to generate only the correlations between each feature\
        and the label.

    method: {'pearson', 'spearman', 'kendall'}, default 'pearson'
        * pearson: measures linear relationships and requires normally distributed and homoscedastic data.
        * spearman: ranked/ordinal correlation, measures monotonic relationships.
        * kendall: ranked/ordinal correlation, measures monotonic relationships. Computationally more expensive but
                   more robust in smaller dataets than 'spearman'.

    cmap: matplotlib colormap name or object, or list of colors, default 'BrBG'
        The mapping from data values to color space.

    figsize: tuple, default (12, 10)
        Use to control the figure size.

    annot: bool, default True
        Use to show or hide annotations.

    dev: bool, default False
        Display figure settings in the plot by setting dev = True. If False, the settings are not displayed.s

    **kwargs: optional
        Additional elements to control the visualization of the plot, e.g.:

        * mask: bool, default True
        If set to False the entire correlation matrix, including the upper triangle is shown. Set dev = False in this \
        case to avoid overlap.
        * vmax: float, default is calculated from the given correlation coefficients.
        Value between -1 or vmin <= vmax <= 1, limits the range of the colorbar.
        * vmin: float, default is calculated from the given correlation coefficients.
        Value between -1 <= vmin <= 1 or vmax, limits the range of the colorbar.
        * linewidths: float, default 0.5
        Controls the line-width inbetween the squares.
        * annot_kws: dict, default {'size' : 10}
        Controls the font size of the annotations. Only available when annot = True.
        * cbar_kws: dict, default {'shrink': .95, 'aspect': 30}
        Controls the size of the colorbar.
        * Many more kwargs are available, i.e. 'alpha' to control blending, or options to adjust labels, ticks ...

        Kwargs can be supplied through a dictionary of key-value pairs (see above).

    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    '''

    # Validate Inputs
    _validate_input_0_1(threshold, 'threshold')
    _validate_input_bool(annot, 'annot')
    _validate_input_bool(dev, 'dev')

    data = pd.DataFrame(data)

    # Obtain correlations
    if isinstance(target, (str, list, pd.Series, np.ndarray)):
        target_data = []
        if isinstance(target, str):
            target_data = data[target]
            data = data.drop(target, axis=1)

        elif isinstance(target, (list, pd.Series, np.ndarray)):
            target_data = pd.Series(target)

        corr = pd.DataFrame(data.corrwith(target_data))
        corr.rename_axis(target, axis=1, inplace=True)
        corr = _corr_selector(corr, split=split, threshold=threshold)
        corr = corr.sort_values(corr.columns[0], ascending=False)
        vmax = np.round(np.nanmax(corr)-0.05, 2)
        vmin = np.round(np.nanmin(corr)+0.05, 2)
        mask = False
        square = False

    else:
        corr = corr_mat(data, split=split, threshold=threshold, method=method).data

        mask = np.triu(np.ones_like(corr, dtype=np.bool))  # Generate mask for the upper triangle
        square = True

        vmax = np.round(np.nanmax(corr.where(~mask))-0.05, 2)
        vmin = np.round(np.nanmin(corr.where(~mask))+0.05, 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Specify kwargs for the heatmap
    kwargs = {'mask': mask,
              'cmap': cmap,
              'annot': annot,
              'vmax': vmax,
              'vmin': vmin,
              'linewidths': .5,
              'annot_kws': {'size': 10},
              'cbar_kws': {'shrink': .95, 'aspect': 30},
              **kwargs}

    # Draw heatmap with mask and some default settings
    sns.heatmap(corr,
                center=0,
                square=square,
                fmt='.2f',
                **kwargs
                )

    ax.set_title(f'Feature-correlation ({method})', fontdict={'fontsize': 18})

    # Display settings
    if dev:
        fig.suptitle(f"\
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
                     color='gray',
                     x=0.35,
                     y=0.85,
                     ha='left')

    return ax


# Distribution plot
def dist_plot(data, mean_color='orange', figsize=(14, 2), fill_range=(0.025, 0.975), hist=False, bins=None,
              showall=False, kde_kws={}, rug_kws={}, fill_kws={}, font_kws={}):
    '''
    Two-dimensional visualization of the distribution of numerical features.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame is provided, the index/column \
    information is used to label the plots.

    mean_color: any valid color, default 'orange'
        Color of the vertical line indicating the mean of the data.

    figsize: tuple, default (14, 2)
        Use to control the figure size.

    fill_range: tuple, default (0.025, 0.975)
        Use to control set the quantiles for shading. Default spans 95% of the data, which is about two std. deviations\
        above and below the mean.

    hist: bool, default False
        Set to True to display histogram bars in the plot.

    bins: integer, default None
        Specification of the number of hist bins. Requires hist = True

    showall: bool, default False
        Set to True to remove the output limit of 20 plots.

    kdw_kws: dict, optional
        Keyword arguments for kdeplot().

    rug_kws: dict, optional
        Keyword arguments for rugplot().

    fill_kws:
        Keyword arguments to control the fill.

    font_kws:
        Keyword arguments to control the font.

    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    '''

    # Validate Inputs
    _validate_input_bool(hist, 'hist')
    _validate_input_bool(showall, 'showall')

    data = pd.DataFrame(data).copy()
    cols = list(data.select_dtypes(include=['number']).columns)  # numeric cols

    if len(cols) == 0:
        print('No columns with numeric data were detected.')
    elif len(cols) >= 20 and showall is False:
        print(
            f'Note: The number of numerical features is very large ({len(cols)}), please consider splitting the data.\
            Showing plots for the first 20 numerical features. Override this by setting showall=True.')
        cols = cols[:20]

    # Default settings
    kde_kws = {'color': 'k', 'alpha': 0.7, 'linewidth': 1, **kde_kws}
    rug_kws = {'color': 'brown', 'alpha': 0.5, 'linewidth': 2, 'height': 0.04, **rug_kws}
    fill_kws = {'color': 'brown', 'alpha': 0.1, **fill_kws}
    font_kws = {'color':  '#111111', 'weight': 'normal', 'size': 11, **font_kws}

    ax = []
    for col in cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.distplot(data[col], bins=bins, hist=hist, rug=True, kde_kws=kde_kws,
                          rug_kws=rug_kws, hist_kws={'alpha': 0.5, 'histtype': 'step'})

        # Vertical lines and fill
        line = ax.lines[0]
        x = line.get_xydata()[:, 0]
        y = line.get_xydata()[:, 1]
        ax.fill_between(x, y,
                        where=(
                            (x >= np.quantile(data[col], fill_range[0])) &
                            (x <= np.quantile(data[col], fill_range[1]))),
                        label=f'{fill_range[0]*100:.0f}% - {fill_range[1]*100:.0f}%',
                        **fill_kws)

        ax.vlines(x=np.mean(data[col]),
                  ymin=0,
                  ymax=np.interp(np.mean(data[col]), x, y),
                  ls='dotted', color=mean_color, lw=2, label='mean')
        ax.vlines(x=np.median(data[col]),
                  ymin=0,
                  ymax=np.interp(np.median(data[col]), x, y),
                  ls=':', color='.3', label='median')
        ax.vlines(x=np.quantile(data[col], 0.25),
                  ymin=0,
                  ymax=np.interp(np.quantile(data[col], 0.25), x, y), ls=':', color='.5', label='25%')
        ax.vlines(x=np.quantile(data[col], 0.75),
                  ymin=0,
                  ymax=np.interp(np.quantile(data[col], 0.75), x, y), ls=':', color='.5', label='75%')

        ax.set_ylim(0,)
        ax.set_xlim(ax.get_xlim()[0]*1.1, ax.get_xlim()[1]*1.1)

        # Annotations and legend
        ax.text(0.01, 0.85, f'Mean: {np.round(np.mean(data[col]),2)}',
                fontdict=font_kws, transform=ax.transAxes)
        ax.text(0.01, 0.7, f'Std. dev: {np.round(scipy.stats.tstd(data[col]),2)}',
                fontdict=font_kws, transform=ax.transAxes)
        ax.text(0.01, 0.55, f'Skew: {np.round(scipy.stats.skew(data[col]),2)}',
                fontdict=font_kws, transform=ax.transAxes)
        ax.text(0.01, 0.4, f'Kurtosis: {np.round(scipy.stats.kurtosis(data[col]),2)}',  # Excess Kurtosis
                fontdict=font_kws, transform=ax.transAxes)
        ax.text(0.01, 0.25, f'Count: {np.round(len(data[col]))}',
                fontdict=font_kws, transform=ax.transAxes)
        ax.legend(loc='upper right')

    return ax


# Missing value plot
def missingval_plot(data, cmap='PuBuGn', figsize=(12, 12), sort=False, spine_color='#EEEEEE'):
    '''
    Two-dimensional visualization of the missing values in a dataset.

    Parameters
    ----------
    data: 2D dataset that can be coerced into Pandas DataFrame. If a Pandas DataFrame is provided, the index/column \
    information is used to label the plots.

    cmap: colormap, default 'PuBuGn'
        Any valid colormap can be used. E.g. 'Greys', 'RdPu'. More information can be found in the matplotlib \
        documentation.

    figsize: tuple, default (20, 12)
        Use to control the figure size.

    sort: bool, default False
        Sort columns based on missing values in descending order and drop columns without any missing values

    spine_color: color-code, default '#EEEEEE'
        Set to 'None' to hide the spines on all plots or use any valid matplotlib color argument.

    Returns
    -------
    figure
    '''

    data = pd.DataFrame(data)

    if sort:
        mv_cols_sorted = data.isna().sum(axis=0).sort_values(ascending=False)
        final_cols = mv_cols_sorted.drop(mv_cols_sorted[mv_cols_sorted.values == 0].keys().tolist()).keys().tolist()
        data = data[final_cols]
        print('Displaying only columns with missing values.')

    # Identify missing values
    mv_cols = _missing_vals(data)['mv_cols']  # data.isna().sum(axis=0)
    mv_rows = _missing_vals(data)['mv_rows']  # data.isna().sum(axis=1)
    mv_total = _missing_vals(data)['mv_total']
    mv_cols_ratio = _missing_vals(data)['mv_cols_ratio']  # mv_cols / data.shape[0]
    total_datapoints = data.shape[0]*data.shape[1]

    if mv_total == 0:
        print('No missing values found in the dataset.')
    else:
        # Create figure and axes
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=6, ncols=6, left=0.05, wspace=0.05)
        ax1 = fig.add_subplot(gs[:1, :5])
        ax2 = fig.add_subplot(gs[1:, :5])
        ax3 = fig.add_subplot(gs[:1, 5:])
        ax4 = fig.add_subplot(gs[1:, 5:])

        # ax1 - Barplot
        colors = plt.get_cmap(cmap)(mv_cols / np.max(mv_cols))  # color bars by height
        ax1.bar(range(len(mv_cols)), np.round((mv_cols_ratio)*100, 2), color=colors)
        ax1.get_xaxis().set_visible(False)
        ax1.set(frame_on=False, xlim=(-.5, len(mv_cols)-0.5))
        ax1.set_ylim(0, np.max(mv_cols_ratio)*100)
        ax1.grid(linestyle=':', linewidth=1)
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        ax1.tick_params(axis='y', colors='#111111', length=1)

        # annotate values on top of the bars
        for rect, label in zip(ax1.patches, mv_cols):
            height = rect.get_height()
            ax1.text(.1 + rect.get_x() + rect.get_width() / 2, height+0.5, label,
                     ha='center',
                     va='bottom',
                     rotation='90',
                     alpha=0.5,
                     fontsize='small')

        ax1.set_frame_on(True)
        for _, spine in ax1.spines.items():
            spine.set_visible(True)
            spine.set_color(spine_color)
        ax1.spines['top'].set_color(None)

        # ax2 - Heatmap
        sns.heatmap(data.isna(), cbar=False, cmap='binary', ax=ax2)
        ax2.set_yticks(np.round(ax2.get_yticks()[0::5], -1))
        ax2.set_yticklabels(ax2.get_yticks())
        ax2.set_xticklabels(
            ax2.get_xticklabels(),
            horizontalalignment='center',
            fontweight='light',
            fontsize='medium')
        ax2.tick_params(length=1, colors='#111111')
        for _, spine in ax2.spines.items():
            spine.set_visible(True)
            spine.set_color(spine_color)

        # ax3 - Summary
        fontax3 = {'color':  '#111111',
                   'weight': 'normal',
                   'size': 12,
                   }
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set(frame_on=False)

        ax3.text(0.1, 0.9, f"Total: {np.round(total_datapoints/1000,1)}K",
                 transform=ax3.transAxes,
                 fontdict=fontax3)
        ax3.text(0.1, 0.7, f"Missing: {np.round(mv_total/1000,1)}K",
                 transform=ax3.transAxes,
                 fontdict=fontax3)
        ax3.text(0.1, 0.5, f"Relative: {np.round(mv_total/total_datapoints*100,1)}%",
                 transform=ax3.transAxes,
                 fontdict=fontax3)
        ax3.text(0.1, 0.3, f"Max-col: {np.round(mv_cols.max()/data.shape[0]*100)}%",
                 transform=ax3.transAxes,
                 fontdict=fontax3)
        ax3.text(0.1, 0.1, f"Max-row: {np.round(mv_rows.max()/data.shape[1]*100)}%",
                 transform=ax3.transAxes,
                 fontdict=fontax3)

        # ax4 - Scatter plot
        ax4.get_yaxis().set_visible(False)
        for _, spine in ax4.spines.items():
            spine.set_color(spine_color)
        ax4.tick_params(axis='x', colors='#111111', length=1)

        ax4.scatter(mv_rows, range(len(mv_rows)), s=mv_rows, c=mv_rows, cmap=cmap, marker=".", vmin=1)
        ax4.set_ylim((0, len(mv_rows))[::-1])  # limit and invert y-axis
        ax4.set_xlim(0, max(mv_rows)+0.5)
        ax4.grid(linestyle=':', linewidth=1)

        gs.figure.suptitle('Missing value plot', x=0.45, y=0.94, fontsize=18, color='#111111')

        return gs
