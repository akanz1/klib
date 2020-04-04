'''
Utilities for data cleaning.

:author: Andreas Kanz

'''

# Imports
import pandas as pd
import statsmodels


# function to drop empty rows and columns based on thresholds and reindex?
# setting for row-wise or colum-wise or both to drop (i.e. might make little sense to drop rows in a time series)

# setting for "hard drop" (if NaN in this field drop row/column) --> Must exist. --> Consider imputation

# list all dropped columns and rows and provide a before and after summary of shape, memory etc


# drop further columns and rows based on criteria


# deal with outliers --> Outlier models? Possible Options? Default values?
# list possible outliers base on standard deviation
# winsorize?
# quantile based
# Dropping the outlier rows with Percentiles
# upper_lim = data['column'].quantile(.95)
# lower_lim = data['column'].quantile(.05)

# capping the data at a certain value
# Capping the outlier rows with Percentiles
# upper_lim = data['column'].quantile(.95)
# lower_lim = data['column'].quantile(.05)
# data.loc[(df[column] > upper_lim),column] = upper_lim
# data.loc[(df[column] < lower_lim),column] = lower_lim


# data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]

# imputation methods
# col or row?
# mean
# median - more outlier resistant
# ...


# further feature engineering

# extract information from datetimes
# create features for year, month, day, weekday, weekend, day of the week, holiday, ...

# from datetime import date

# data = pd.DataFrame({'date':
# ['01-01-2017',
# '04-12-2008',
# '23-06-1988',
# '25-08-1999',
# '20-02-1993',
# ]})

# #Transform string to date
# data['date'] = pd.to_datetime(data.date, format="%d-%m-%Y")

# #Extracting Year
# data['year'] = data['date'].dt.year

# #Extracting Month
# data['month'] = data['date'].dt.month

# #Extracting passed years since the date
# data['passed_years'] = date.today().year - data['date'].dt.year

# #Extracting passed months since the date
# data['passed_months'] = (date.today().year - data['date'].dt.year) * 12 + date.today().month - data['date'].dt.month

# #Extracting the weekday name of the date
# data['day_name'] = data['date'].dt.day_name()
#         date  year  month  passed_years  passed_months   day_name
# 0 2017-01-01  2017      1             2             26     Sunday
# 1 2008-12-04  2008     12            11            123   Thursday
# 2 1988-06-23  1988      6            31            369   Thursday
# 3 1999-08-25  1999      8            20            235  Wednesday
# 4 1993-02-20  1993      2            26            313   Saturday

# binning (give option to choose features to bin and how)
# binning of numerical variables
# binning of categorical data

# Categorical Binning Example
#      Country
# 0      Spain
# 1      Chile
# 2  Australia
# 3      Italy
# 4     Brazil
# conditions = [
#     data['Country'].str.contains('Spain'),
#     data['Country'].str.contains('Italy'),
#     data['Country'].str.contains('Chile'),
#     data['Country'].str.contains('Brazil')]

# choices = ['Europe', 'Europe', 'South America', 'South America']

# data['Continent'] = np.select(conditions, choices, default='Other')
#      Country      Continent
# 0      Spain         Europe
# 1      Chile  South America
# 2  Australia          Other
# 3      Italy         Europe
# 4     Brazil  South America

# encode dummies from categorical features unsing sklearn one hot (check intercept, regularization etc.) provide description why sklearn instead of pd.get_dummies

# export / save " clean data"?


def convert_datatypes(data, category=True, cat_threshold=0.05, exclude=[]):
    '''
    Convert columns to best possible dtypes using dtypes supporting pd.NA.

    Parameters:
    ----------
    data: Pandas DataFrame or numpy ndarray.

    category: bool, default True
        Change dtypes of columns to "category". Set threshold using cat_threshold.

    cat_threshold: float, default 0.05
        Ratio of unique values below which column dtype is changed to categorical.

    exclude: default [] (empty list)
        List of columns to exclude from categorical conversion.

    Returns:
    -------
    Pandas DataFrame or numpy ndarray.

    '''

    data = pd.DataFrame(data).copy()
    for col in data.columns:
        data[col] = data[col].convert_dtypes()
        unique_vals_ratio = data[col].nunique(dropna=False) / data.shape[0]
        if category and unique_vals_ratio < cat_threshold and col not in exclude:
            data[col] = data[col].astype('category')
    return data


def memory_usage(data):  # --> describe.py & setup imports
    '''
    Total memory usage in kilobytes.

    Parameters:
    ----------
    data: Pandas DataFrame or numpy ndarray.

    Returns:
    -------
    memory_usage: float

    '''

    data = pd.DataFrame(data)
    memory_usage = round(data.memory_usage(index=True, deep=True).sum()/1024, 2)
    return memory_usage


def missing_vals(data):  # --> describe.py & setup imports
    '''
    Different representations of missing values in the dataset.

    Parameters:
    ----------

    data: Pandas DataFrame or numpy ndarray.

    Returns:
    -------
    total_mv: float, number of missing values in the entire dataset
    rows_mv: float, number of missing values in each row
    cols_mv: float, number of missing values in each column
    rows_mv_ratio: float, ratio of missing values for each row
    cols_mv_ratio: float, ratio of missing values for each column
    '''
    rows_mv = data.isna().sum(axis=0)
    cols_mv = data.isna().sum(axis=1)
    total_mv = data.isna().sum().sum()
    rows_mv_ratio = rows_mv/data.shape[0]
    cols_mv_ratio = cols_mv/data.shape[1]
    return total_mv, rows_mv, cols_mv, rows_mv_ratio, cols_mv_ratio


def drop_missing(data, drop_threshold_cols, drop_threshold_rows):
    '''
    '''
    data = data.dropna(axis=0, how='all')
    data = data.dropna(axis=1, how='all')
    data = data.drop(columns=data.loc[:, missing_vals(data)[3] > drop_threshold_cols].columns)  # drop cols
    data_cleaned = data.drop(index=data.loc[missing_vals(data)[4] > drop_threshold_rows, :].index)  # drop rows
    return data_cleaned


# def summary_stats(data):  # --> describe.py & setup imports --> focus on distribution of column(s)
#     rows = data.shape[0]
#     cols = data.shape[1]
#     # look for possible statistics (statsmodels)
#     # no visualizations here
#     # Shape
#     return rows, cols


def data_cleaning(data, drop_threshold_cols=0.75, drop_threshold_rows=0.75, category=True, cat_threshold=0.05, exclude=[], show='all'):
    '''
    initial data cleaning


    Note: The category dtype is not grouped in the summary, unless it contains exactly the same categories.
    '''

    print('Dropping empty rows / columns and subsequently rows and columns with missing values above the specified thresholds ...')

    data_cleaned = drop_missing(data, drop_threshold_cols, drop_threshold_rows)

    print('Adjusting dtypes ...')
    print('Done.')
    data_cleaned = convert_datatypes(data_cleaned, category=True, cat_threshold=0.05, exclude=exclude)  # change dtypes
    if show == 'changes' or 'all':
        if show == 'all':
            print('\nBefore data cleaning:\n')
            print(f'dtypes:\n{data.dtypes.value_counts()}')
            print(f'\nNumber of rows: {data.shape[0]}')
            print(f'Number of cols: {data.shape[1]}')
            print(f'Missing values: {missing_vals(data)[0]}')
            print(f'Memory usage: {memory_usage(data)} KB')
            print('_______________________\n')
            print('After data cleaning:\n')
            print(f'dtypes:\n{data_cleaned.dtypes.value_counts()}')
            print(f'\nNumber of rows: {data_cleaned.shape[0]}')
            print(f'Number of cols: {data_cleaned.shape[1]}')
            print(f'Missing values: {missing_vals(data_cleaned)[0]}')
            print(f'Memory usage: {memory_usage(data_cleaned)} KB')
        print('_______________________\n')
        print(f'Changes:\n')
        print(f'Dropped rows: {data.shape[0]-data_cleaned.shape[0]}')
        print(f'Dropped columns: {data.shape[1]-data_cleaned.shape[1]}')
        print(f'Dropped missing values: {missing_vals(data)[0]-missing_vals(data_cleaned)[0]}')
        mem_change = memory_usage(data)-memory_usage(data_cleaned)
        print(f'Reduced memory by: {mem_change} KB (-{round(100*mem_change/memory_usage(data),1)}%)')

    return data_cleaned
