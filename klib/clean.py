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
    
    
    '''
    data = data.copy()
    for col in data.columns:
        data[col] = data[col].convert_dtypes()
        unique_vals_ratio = data[col].nunique(dropna=False) / data.shape[0]
        if category and unique_vals_ratio < cat_threshold and col not in exclude:
            data[col] = data[col].astype('category')
    return data


def memory_usage(data):
    '''
    Total memory usage in Kilobytes.
    '''
    memory_usage = round(data.memory_usage(index=True, deep=True).sum()/1024, 2)
    return memory_usage
