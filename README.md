<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/header.png" alt="klib Header" width="859" height="304"></p>

[![Flake8 & PyTest](https://github.com/akanz1/klib/workflows/Flake8%20%F0%9F%90%8D%20PyTest%20%20%20%C2%B4/badge.svg)](https://github.com/akanz1/klib)
[![Language](https://img.shields.io/github/languages/top/akanz1/klib)](https://pypi.org/project/klib/)
[![Last Commit](https://badgen.net/github/last-commit/akanz1/klib/main)](https://github.com/akanz1/klib/commits/main)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=akanz1_klib&metric=alert_status)](https://sonarcloud.io/dashboard?id=akanz1_klib)
[![Scrutinizer](https://scrutinizer-ci.com/g/akanz1/klib/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/akanz1/klib/)
[![codecov](https://codecov.io/gh/akanz1/klib/branch/main/graph/badge.svg)](https://codecov.io/gh/akanz1/klib)

**klib** is a Python library for importing, cleaning, analyzing and preprocessing data. Explanations on key functionalities can be found on [Medium / TowardsDataScience](https://medium.com/@akanz) and in the [examples](examples) section. Additionally, there are great introductions and overviews of the functionality on [PythonBytes](https://pythonbytes.fm/episodes/show/240/this-is-github-your-pilot-speaking) or on [YouTube (Data Professor)](https://www.youtube.com/watch?v=URjJVEeZxxU).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install klib.

[![PyPI Version](https://img.shields.io/pypi/v/klib)](https://pypi.org/project/klib/)
[![Downloads](https://pepy.tech/badge/klib/month)](https://pypi.org/project/klib/)

```bash
pip install -U klib
```

Alternatively, to install this package with conda run:

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/klib)](https://anaconda.org/conda-forge/klib)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/klib.svg)](https://anaconda.org/conda-forge/klib)

```bash
conda install -c conda-forge klib
```

## Usage

```python
import klib
import pandas as pd

df = pd.DataFrame(data)

# klib.describe - functions for visualizing datasets
- klib.cat_plot(df) # returns a visualization of the number and frequency of categorical features
- klib.corr_mat(df) # returns a color-encoded correlation matrix
- klib.corr_plot(df) # returns a color-encoded heatmap, ideal for correlations
- klib.dist_plot(df) # returns a distribution plot for every numeric feature
- klib.missingval_plot(df) # returns a figure containing information about missing values

# klib.clean - functions for cleaning datasets
- klib.data_cleaning(df) # performs datacleaning (drop duplicates & empty rows/cols, adjust dtypes,...)
- klib.clean_column_names(df) # cleans and standardizes column names, also called inside data_cleaning()
- klib.convert_datatypes(df) # converts existing to more efficient dtypes, also called inside data_cleaning()
- klib.drop_missing(df) # drops missing values, also called in data_cleaning()
- klib.mv_col_handling(df) # drops features with high ratio of missing vals based on informational content
- klib.pool_duplicate_subsets(df) # pools subset of cols based on duplicates with min. loss of information
```

## Examples

Find all available examples as well as applications of the functions in **klib.clean()** with detailed descriptions <a href="https://github.com/akanz1/klib/tree/main/examples">here</a>.

```python
klib.missingval_plot(df) # default representation of missing values in a DataFrame, plenty of settings are available
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_mv_plot.png" alt="Missingvalue Plot Example" width="1000" height="1091"></p>

```python
klib.corr_plot(df, split='pos') # displaying only positive correlations, other settings include threshold, cmap...
klib.corr_plot(df, split='neg') # displaying only negative correlations
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_corr_plot.png" alt="Corr Plot Example" width="720" height="338"></p>

```python
klib.corr_plot(df, target='wine') # default representation of correlations with the feature column
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_target_corr_plot.png" alt="Target Corr Plot Example" width="720" height="600"></p>

```python
klib.corr_interactive_plot(df, split="neg").show()

# The interactive plot has the same parameters as the corr_plot, but with additional Plotly heatmap graph object kwargs.
klib.corr_interactive_plot(df, split="neg", zmax=0)
```

<p align="center"><img src="https://github.com/akanz1/klib/assets/124513922/27b79ece-dc22-410c-9816-fcc4cfc97edc"  alt="Interactive Corr Plot Simple Example" width="720" height="600"></p>

<p align="center"><img src="https://github.com/akanz1/klib/assets/124513922/d986dc6c-84dc-4693-b276-6b4df7023e0f" alt="Interactive Corr Plot with zmax kwarg Example" width="720" height="600"></p>

```python
#Since corr_interactive_plot returns a Graph Object Figure, it supports the update_layout chain method.
klib.corr_interactive_plot(wine, split="neg").update_layout(template="simple_white")
```

<p align="center"><img src="https://github.com/akanz1/klib/assets/124513922/fb7ac102-6daf-4e30-a24c-db168c7cf42c" alt="Interactive Corr Plot Chained Example" width="720" height="600"></p>

```python
klib.dist_plot(df) # default representation of a distribution plot, other settings include fill_range, histogram, ...
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_dist_plot.png" alt="Dist Plot Example" width="910" height="130"></p>

```python
klib.cat_plot(data, top=4, bottom=4) # representation of the 4 most & least common values in each categorical column
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_cat_plot.png" alt="Cat Plot Example" width="1000" height="1000"></p>

Further examples, as well as applications of the functions in **klib.clean()** can be found <a href="https://github.com/akanz1/klib/tree/main/examples#data-cleaning-and-aggretation">here</a>.

## Contributing

[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/akanz1/klib)

Pull requests and ideas, especially for further functions are welcome. For major changes or feedback, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
