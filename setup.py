import setuptools

from klib import _version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="klib",
    version=f"{_version.__version__}",
    author="Andreas Kanz",
    author_email="andreas@akanz.de",
    description="Customized data preprocessing functions for frequent tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akanz1/klib",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "matplotlib >= 3.0.3",
        "numpy >= 1.15.4",
        "pandas >= 1.1",
        "seaborn >= 0.11.1",
        "scikit-learn >= 0.23",
        "scipy >= 1.0.0",
    ],
    python_requires=">=3.7",
)
