'''
Utilities for descriptive analytics.

:author: Andreas Kanz

'''

# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Correlation matrix / heatmap
def corr_plot(data, split=None, threshold=0.5, dev=False, **kwargs):
    '''
    Two-dimensional visualization of the correlation between feature-columns, excluding NA values.

    Parameters:
    ----------
    data: 2D dataset that can be coerced into an ndarray. If a Pandas DataFrame is provided, the index/column information will be used to label the columns and rows.

    split: {'None', 'pos', 'neg', 'high', 'low'}, default 'None'
        Type of split to be performed.

        * None: visualize all correlations between the feature-columns.
        * pos: visualize all positive correlations between the feature-columns above the threshold.
        * neg: visualize all negative correlations between the feature-columns below the threshold.
        * high: visualize all correlations between the feature-columns for which abs(corr) > threshold is True.
        * low: visualize all correlations between the feature-columns for which abs(corr) < threshold is True.

    threshold: float, default 0.5
        Value between 0 <= threshold <= 1

    dev: bool, default False
        Display figure settings in the plot by setting dev = True. If False, the settings are not displayed. Use for presentations.

    **kwargs:
        Additional elements to control the visualization of the plot, e.g.:

        * mask: bool, default True
        If set to False the entire correlation matrix, including the upper triangle is shown. Set dev = False in this case to avoid overlap.
        * cmap: matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default is sns.color_palette("BrBG", 150).
        * annot:bool, default True for 20 or less columns, False for more than 20 feature-columns.
        * vmax: float, default is calculated from the given correlation coefficients. 
        Value between -1 or vmin <= vmax <= 1, limits the range of the colorbar.
        * vmin: float, default is calculated from the given correlation coefficients. 
        Value between -1 <= vmin <= 1 or vmax, limits the range of the colorbar.
        * linewidths: float, default 0.5
        Controls the line-width inbetween the squares.
        * annot_kws: dict, default {'size' : 10}
        Controls the font size of the annotations. Only available when annot = True.
        * cbar_kws: dict, default {'shrink': .8}
        Controls the size of the colorbar.
        * Many more kwargs are available, i.e. 'alpha' to control blending, or options to adjust labels, ticks ...

        Kwargs can be supplied through a dictionary of key-value pairs (see above).

    Returns:
    ------- 
    ax: matplotlib Axes. Axes object with the heatmap.
    '''

    if split == 'pos':
        corr = data.corr().where((data.corr() >= threshold) & (data.corr()>0))
        threshold = '-'
    elif split == 'neg':
        corr = data.corr().where((data.corr() <= threshold) & (data.corr()<0))
        threshold = '-'
    elif split == 'high':
        corr = data.corr().where(np.abs(data.corr()) >= threshold)
    elif split == 'low':
        corr = data.corr().where(np.abs(data.corr()) <= threshold)
    else:
        corr = data.corr()
        split = "full"
        threshold = 'None'

    # Generate mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Compute dimensions and correlation range to adjust settings
    annot = True if np.max(corr.shape) < 21 else False
    vmax = np.round(np.nanmax(corr.where(mask == False))-0.05, 2)
    vmin = np.round(np.nanmin(corr.where(mask == False))+0.05, 2)

    # Set up the matplotlib figure and generate colormap
    if np.max(corr.shape) < 11:
        fsize = (8, 6)
    elif np.max(corr.shape) < 16:
        fsize = (10, 8)
    else:
        fsize = (12, 10)
    fig, ax = plt.subplots(figsize=fsize)
    cmap = sns.color_palette("BrBG", 150)

    # Draw heatmap with mask and correct aspect ratio
    kwargs = {'mask': mask,
              'cmap': cmap,
              'annot': annot,
              'vmax': vmax,
              'vmin': vmin,
              'linewidths': .5,
              'annot_kws': {'size': 10},
              'cbar_kws': {'shrink': .8},
              **kwargs}

    sns.heatmap(corr,
                center=0,
                square=True,
                fmt='.2f',
                **kwargs
                )

    ax.set_title('Feature-correlation Matrix', fontdict={'fontsize': 18})

    if dev == False:
        pass
    else:  # show settings
        fig.suptitle(f"\
            Settings (dev-mode): \n\
            - Split-mode: {split} \n\
            - Threshold: {threshold} \n\
            - CBar: \n\
                - vmax: {vmax} \n\
                - vmin: {vmin} \n\
            - linewidths: {kwargs['linewidths']} \n\
            - annot_kws: {kwargs['annot_kws']} \n\
            - cbar_kws: {kwargs['cbar_kws']}",
                     fontsize=12,
                     color='gray',
                     x=0.5,
                     y=0.75,
                     ha='left')
