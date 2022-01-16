#############################
Installation and contribution
#############################

Prerequisites
=============

klib requires the following dependencies:

* matplotlib >= 3.0.3
* numpy >= 1.15.4
* pandas >= 1.1
* python >= 3.7
* scikit-learn >= 0.23
* scipy >= 1.0.0
* seaborn >= 0.11.1

Install
=======
Use the package manager `pip <https://pip.pypa.io/en/stable/>`__ to
install klib:

|PyPI Version|

.. code:: bash

    pip install klib
    pip install --upgrade klib

Alternatively, to install this package with conda run:

|Conda Version|

.. code:: bash

    conda install -c conda-forge klib

You can also fork/clone the repository and run the setup.py file. Use the following commands to get a copy from Github and install all dependencies::

  git clone https://github.com/akanz1/klib.git
  cd into package root dir
  pip install .

Or install directly from GitHub using pip::

  pip install -U git+https://github.com/akanz1/klib.git

Contribute
==========

Pull requests and ideas, especially for further functions and visualizations are welcome. For major changes or feedback, please open an issue first to discuss what you would like to change.


.. |PyPI Version| image:: https://img.shields.io/pypi/v/klib
   :target: https://pypi.org/project/klib/
.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/klib
   :target: https://anaconda.org/conda-forge/klib
