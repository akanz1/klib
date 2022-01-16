klib
====

| |Flake8 & PyTest|
| |Language|
| |Downloads|
| |Last Commit|
| |Quality Gate Status|
| |Scrutinizer|

klib is a Python library for importing, cleaning, analyzing and
preprocessing data. Future versions will include model creation and
optimization to provide an end-to-end solution.

Installation
------------

Use the package manager `pip <https://pip.pypa.io/en/stable/>`__ to
install klib.

|PyPI Version|

.. code:: bash

    pip install klib
    pip install --upgrade klib

Alternatively, to install this package with conda run:

|Conda Version|

.. code:: bash

    conda install -c conda-forge klib

Usage
-----

.. code:: python

    import klib

    klib.describe # functions for visualizing datasets
    - klib.cat_plot() # returns a visualization of the number and frequency of categorical features.
    - klib.corr_mat() # returns a color-encoded correlation matrix
    - klib.corr_plot() # returns a color-encoded heatmap, ideal for correlations
    - klib.dist_plot() # returns a distribution plot for every numeric feature
    - klib.missingval_plot() # returns a figure containing information about missing values

    klib.clean # functions for cleaning datasets
    - klib.data_cleaning() # performs datacleaning (drop duplicates & empty rows/columns, adjust dtypes,...) on a dataset
    - klib.convert_datatypes() # converts existing to more efficient dtypes, also called inside ".data_cleaning()"
    - klib.drop_missing() # drops missing values, also called in ".data_cleaning()"
    - klib.mv_col_handling() # drops features with a high ratio of missing values based on their informational content
    - klib.pool_duplicate_subsets() # pools a subset of columns based on duplicate values with minimal loss of information

    klib.preprocess # functions for data preprocessing (feature selection, scaling, ...)
    - klib.train_dev_test_split() # splits a dataset and a label into train, optionally dev and test sets
    - klib.feature_selection_pipe() # provides common operations for feature selection
    - klib.num_pipe() # provides common operations for preprocessing of numerical data
    - klib.cat_pipe() # provides common operations for preprocessing of categorical data
    - klib.preprocess.ColumnSelector() # selects numerical or categorical columns, ideal for a Feature Union or Pipeline
    - klib.preprocess.PipeInfo() # prints out the shape of the data at the specified step of a Pipeline

Examples
--------

Find all available examples as well as applications of the functions in
**klib.clean()** with detailed descriptions here.

This plot visualizes the missing values in a dataset. At the top it shows the aggregate for each column using a relative scale and absolute missing-value annotations, while on the right, summary statistics and individual row results are displayed. Using this plot allows to gain a quick overview over the structure of missing values and their relation in a dataset and easily determine which columns and rows to investigate / drop.

.. code:: python

    klib.missingval_plot(df) # default representation of missing values, other settings such as sorting are available

.. raw:: html

   <p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_mv_plot.png" alt="Missingvalue Plot Example" width="1100" height="1200"></p>


This plot visualizes the correlation between different features. Settings include the possibility to only display positive, negative, high or low correlations as well as specify an additional threshold. This works for Person, Spearman and Kendall correlation. Annotations and development settings can optionally be turned on or off.

.. code:: python

    klib.corr_plot(df, split='pos') # displaying only positive correlations, other settings include threshold, cmap...
    klib.corr_plot(df, split='neg') # displaying only negative correlations

.. raw:: html

   <p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_corr_plot.png" alt="Corr Plot Example" width="720" height="338"></p>

Further, as seen below, if a column is specified, either by name or by passing in a separate target List or pd.Series, the plot gives the correlation of all features with the specified target.

.. code:: python

    klib.corr_plot(df, target='wine') # default representation of correlations with the feature column

.. raw:: html

   <p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_target_corr_plot.png" alt="Target Corr Plot Example" width="720" height="600"></p>


.. code:: python

    klib.dist_plot(df) # default representation of a distribution plot, other settings include fill_range, histogram, ...

.. raw:: html

   <p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_dist_plot.png" alt="Dist Plot Example" width="910" height="130"></p>


.. code:: python

    klib.cat_plot(data, top=4, bottom=4) # representation of the 4 most & least common values in each categorical column

.. raw:: html

   <p align="center"><img src="https://raw.githubusercontent.com/akanz1/klib/main/examples/images/example_cat_plot.png" alt="Cat Plot Example" width="1000" height="1000"></p>

Further examples, as well as applications of the functions in
**klib.clean()** can be found here.

Contributing
------------

Pull requests and ideas, especially for further functions are welcome.
For major changes or feedback, please open an issue first to discuss
what you would like to change.

License
-------

`MIT <https://choosealicense.com/licenses/mit/>`__

.. |Flake8 & PyTest| image:: https://github.com/akanz1/klib/workflows/Flake8%20%F0%9F%90%8D%20PyTest%20%20%20%C2%B4/badge.svg
   :target: https://github.com/akanz1/klib
.. |Language| image:: https://img.shields.io/github/languages/top/akanz1/klib
   :target: https://pypi.org/project/klib/
.. |Downloads| image:: https://img.shields.io/pypi/dm/klib
   :target: https://pypi.org/project/klib/
.. |Last Commit| image:: https://badgen.net/github/last-commit/akanz1/klib/main
   :target: https://github.com/akanz1/klib/commits/main
.. |Quality Gate Status| image:: https://sonarcloud.io/api/project_badges/measure?project=akanz1_klib&metric=alert_status
   :target: https://sonarcloud.io/dashboard?id=akanz1_klib
.. |Scrutinizer| image:: https://scrutinizer-ci.com/g/akanz1/klib/badges/quality-score.png?b=main
   :target: https://scrutinizer-ci.com/g/akanz1/klib/
.. |PyPI Version| image:: https://img.shields.io/pypi/v/klib
   :target: https://pypi.org/project/klib/
.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/klib
   :target: https://anaconda.org/conda-forge/klib
