# klib

klib is a Python library for importing, cleaning, analyzing and preprocessing data. Future versions will include model creation and optimization to provide an end-to-end solution.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install klib.

```bash
pip install klib
```

## Usage

```python
import klib

# klib.ingest() # runs ...

klib.corr_plot() # returns a color-encoded matrix, ideal for correlations
klib.missingval_plot() # returns a figure containing information about missing values

# klib.clean() # does ...
# klib.preprocess() # yields ...
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)