# klib

[![Python package](https://github.com/akanz1/klib/workflows/Python%20package/badge.svg)](https://github.com/akanz1/klib)
[![PyPI Version](https://img.shields.io/pypi/v/klib)](https://pypi.org/project/klib/)
[![Language](https://img.shields.io/github/languages/top/akanz1/klib)](https://pypi.org/project/klib/)
[![Downloads](https://img.shields.io/pypi/dw/klib)](https://pypi.org/project/klib/)
[![Last Commit](https://img.shields.io/github/last-commit/akanz1/klib)](https://github.com/akanz1/klib)
[![Activity](https://img.shields.io/github/commit-activity/m/akanz1/klib)](https://github.com/akanz1/klib)
[![Code Quality](https://scrutinizer-ci.com/g/akanz1/klib/badges/quality-score.png?b=master)](https://github.com/akanz1/klib)
[![License](https://img.shields.io/pypi/l/klib)](https://github.com/akanz1/klib/blob/master/LICENSE)

klib is a Python library for importing, cleaning, analyzing and preprocessing data. Future versions will include model creation and optimization to provide an end-to-end solution.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install klib.

```bash
pip install klib
pip install --upgrade klib
```

## Usage

```python
import klib

klib.describe # tools for visualizing datasets
- klib.corr_mat() # returns acolor-encoded correlation matrix
- klib.corr_plot() # returns a color-encoded heatmap, ideal for correlations
- klib.missingval_plot() # returns a figure containing information about missing values

klib.clean # tools for cleaning datasets
- klib.data_cleaning() # perform initial datacleaning on a dataset
- klib.convert_datatypes() # converts existing to more efficient dtypes, also called in ".data_cleaning()"
- klib.drop_missing() # drops missing values, also called in ".data_cleaning()"
```

## Examples

```python
klib.corr_plot(df, split='pos') # displaying only positive correlations
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/images/example_corr_plot.png" alt="Corr Plot Example" width="720" height="655"></p>

```python
klib.missingval_plot(df) # default representation of missing values in a DataFrame
```

<p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/master/images/example_mv_plot.png" alt="Corr Plot Example" width="720" height="792"></p>

## Contributing

Pull requests and ideas are welcome. For major changes or feedback, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
