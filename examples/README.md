# klib

## Examples

### Missing Value Plot

This plot visualizes the missing values in a dataset. At the top it shows the aggregate for each column using a relative scale and absolute missing-value annotations, while on the right, summary statistics and individual row results are displayed.

```python
klib.missingval_plot(df) # default representation of missing values, other settings such as sorting are available
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_mv_plot.png" alt="Corr Plot Example" width="792" height="970"></p>

### Correlation Plots

This plot visualizes the correlation between different features. Settings include the possibility to only display positive, negative, high or low correlations as well as specify an additional threshold. This works for Person, Spearmann and Kendall correlation. Annotations and development settings can optionally be turned on or off.

```python
klib.corr_plot(df, split='pos') # displaying only positive correlations, other settings include threshold, different colormap...
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_corr_plot.png" alt="Corr Plot Example" width="792" height="721"></p>

Further, as seen below, if a column is specified, either by name or by passing in a separate target List or Series, the plot gives the correlation of all features with the specified target.

```python
klib.corr_plot(df, target='air_time') # default representation of correlations with the feature column
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_target_corr_plot.png" alt="Target Corr Plot Example" width="792" height="660"></p>

```python
klib.corr_plot(df, target='air_time') # default representation of a correlation matrix
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_klib_corr_mat.png" alt="Target Corr Plot Example" width="808" height="369"></p>

### Numerical Data Distribution Plot

```python
klib.dist_plot(df) # default representation of a distribution plot, other settings include fill_range, histogram, ...
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_dist_plot.png" alt="Dist Plot Example" width="770" height="110"></p>

### Categorical Data Plot

This section shows an example of categorical data visualization. The function allows to dispaly the top and/or bottom values regarding their frequency in each column. Further, it gives an idea of the distribution of the values in the dataset.

```python
klib.cat_plot(data, top=4, bottom=4) # representation of the 4 most & least common values in each categorical column
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_cat_plot.png" alt="Cat Plot Example" width="900" height="900"></p>

### Data Cleaning and Aggretation

This sections describes the data cleaning and aggregation capabilities of <a href="https://github.com/akanz1/klib/">klib</a>. We start with an initial dataset about US flight data, which has a size of about 51.5 MB.

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_klib_pool_duplicate_subsets3.png" alt="Original Dataset" width="329" height="376"></p>

After applying *klib.data_cleaning()* **the size reduces by about 36 MB (-69.2%)**. This is achieved by dropping empty and single valued columns as well as empty and duplicate rows (neither found in this example). Additionally, the optimal data types are inferred and applied, which also increases memory efficiency.

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_klib_data_cleaning_dtypes.png" alt="Change in dtypes" width="294" height="440"></p>

Further, *pool_duplicate_subsets()* can be applied to aggregate the columns. **This ultimately reduces the dataset to only 6.8 MB (from 55.1 MB originally)**.

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_klib_pool_duplicate_subsets2.png" alt="Duplicate subsets" width="571" height="469"></p>

The columns are "pooled" **without loss of information**. This can be achieved by finding duplicates in subsets of the data and encoding the largest possible subset with integers, which are added to the original data what allows dropping the identified columns.

As can be seen in <a href="https://github.com/akanz1/klib/tree/master/examples#categorical-data-plot">*cat_plot()*</a> the "carrier" column is made up of a few very frequent values - the top 4 values make up roughly 75% - while in "tailnum" the top 4 values barely make up 2%. This allows "carrier" and similar columns to be bundled and encoded, while "tailnum" remains in the dataset. Using this procedure, 56006 duplicate rows are identified in the subset, i.e., **56006 rows in 10 columns are encoded in a single column of dtype integer**, greatly reducing the memory footprint and number of columns which should speed up model training.

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/examples/images/example_klib_pool_duplicate_subsets1.png" alt="Duplicate subsets2" width="945" height="424"></p>

All of these functions were run with default settings but many parameters are available allowing an even more restrictive data cleaning.
