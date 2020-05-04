# klib

## Examples

### Missing Value Plot

This plot visualizes the missing values in a dataset. At the top it shows the aggregate for each column using a relative scale and absolute missing-value annotations, while on the right, summary statistics and individual row results are displayed.

```python
klib.missingval_plot(df) # default representation of missing values in a DataFrame, plenty of settings are available
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_mv_plot.png" alt="Corr Plot Example" width="720" height="864"></p>

### Correlation Plots

This plot visualizes the correlation between different features. Settings include the possibility to only display positive, negative, high or low correlations as well as specify an additional threshold. THi works for person, spearmann and kendall correlation. Annotations and development settings can optionally be turned on or off.

```python
klib.corr_plot(df, split='pos') # displaying only positive correlations
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_corr_plot.png" alt="Corr Plot Example" width="720" height="656"></p>

Further, as seen below, if a column is specified, either by name or by passing in a separate target List or Series, the plot gives the correlation of all features with the specified target.

```python
klib.corr_plot(df, target='air_time') # default representation of correlations with the feature column
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_target_corr_plot.png" alt="Target Corr Plot Example" width="720" height="600"></p>

### Categorical Data Plot

```python
klib.cat_plot(data, top=4, bottom=4) # representation of the 4 most & least common values in each categorical column
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_cat_plot.png" alt="Cat Plot Example" width="720" height="720"></p>
