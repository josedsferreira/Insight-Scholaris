import pandas as pd
from sklearn.preprocessing import LabelEncoder

gender_mapping = {'M': 0, 'F': 1, 'Other': 2}

disability_mapping = {'N': 0, 'Y': 1}

final_result_mapping = {'Pass': 1, 'Distinction': 1, 'Withdrawn': 0, 'Fail': 0}

age_mapping = {'0-35': 0, '35-55': 1, '55<=': 2}

highest_education_mapping = {'No Formal quals': 0, \
                             'Lower Than A Level': 1, \
                                'A Level or Equivalent': 2, \
                                    'HE Qualification': 3, \
                                        'Post Graduate Qualification': 4}

imd_band_mapping = {'0-10%': 0, '10-20': 1, '20-30%': 2, '30-40%': 3, '40-50%': 4, \
                    '50-60%': 5, '60-70%': 6, '70-80%': 7, '80-90%': 8, '90-100%': 9}

code_presentation_mapping = {'2013B': 0, '2013J': 1, '2014B': 2, '2014J': 3}

code_module_mapping = {'AAA': 0, 'BBB': 1, 'CCC': 2, 'DDD': 3, 'EEE': 4, 'FFF': 5, 'GGG': 6}

region_mapping = {'Scotland': 0, \
                  'East Anglian Region': 1, \
                    'London Region': 2, \
                        'South Region': 3, \
                            'North Western Region': 4, \
                                'West Midlands Region': 5, \
                                    'South West Region': 6, \
                                        'East Midlands Region': 7, \
                                            'Wales': 8, \
                                                'Yorkshire Region': 9, \
                                                    'North Region': 10, \
                                                        'Ireland': 11, \
                                                            'South East Region': 12, \
                                                                'Central Region': 13, \
                                                                    'North Western Region': 14}

reverse_gender_mapping = {v: k for k, v in gender_mapping.items()}
reverse_disability_mapping = {v: k for k, v in disability_mapping.items()}
reverse_final_result_mapping = {v: k for k, v in final_result_mapping.items()}
reverse_age_mapping = {v: k for k, v in age_mapping.items()}
reverse_highest_education_mapping = {v: k for k, v in highest_education_mapping.items()}
reverse_imd_band_mapping = {v: k for k, v in imd_band_mapping.items()}
reverse_code_presentation_mapping = {v: k for k, v in code_presentation_mapping.items()}
reverse_code_module_mapping = {v: k for k, v in code_module_mapping.items()}
reverse_region_mapping = {v: k for k, v in region_mapping.items()}

def clean_data(df):
    """
    Cleans the given DataFrame by applying necessary transformations.

    Args:
        df (pandas.DataFrame): The DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    if is_clean(df):
        return df
    else:
        df = encoder(df)

    return df

def remove_nans(df):
    """
    Removes rows with missing values from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to remove missing values from.

    Returns:
    pandas.DataFrame: The DataFrame with missing values removed.
    """
    return df.dropna()

def remove_nans_columns(df, column_list):
    """
    Remove rows with NaN values in the specified columns from the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to remove rows from.
    column_list (list): A list of column names to check for NaN values.

    Returns:
    pandas.DataFrame: The DataFrame with rows containing NaN values in the specified columns removed.
    """
    return df.dropna(subset=column_list)

def average_nans(df, column_list):
    """
    Fill missing values in the specified columns of a DataFrame with the column's mean.

    Args:
        df (pandas.DataFrame): The DataFrame containing the columns to be processed.
        column_list (list): A list of column names to be processed.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled with the column's mean.
    """
    for column_name in column_list:
        df[column_name].fillna(df[column_name].mean(), inplace=True)
    return df

def mode_nans(df, column_list):
    """
    Fill missing values in the specified columns of a DataFrame with the mode value of each column.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the columns with missing values.
    - column_list (list): A list of column names to fill the missing values.

    Returns:
    - df (pandas.DataFrame): The DataFrame with missing values filled using the mode value of each column.
    """
    for column_name in column_list:
        df[column_name].fillna(df[column_name].mode()[0], inplace=True)
    return df

def remove_columns(df, column_list):
    """
    Removes specified columns from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame from which columns will be removed.
    column_list (list): A list of column names to be removed.

    Returns:
    pandas.DataFrame: The DataFrame with specified columns removed.
    """
    return df.drop(column_list, axis=1)

def map_column(df, column_name, map):
    """
    Maps the values in a specified column of a DataFrame using a given mapping dictionary.

    Args:
        df (pandas.DataFrame): The DataFrame to be modified.
        column_name (str): The name of the column to be mapped.
        map (dict): A dictionary containing the mapping values.

    Returns:
        pandas.DataFrame: The modified DataFrame with the mapped values in the specified column.
    """
    df[column_name] = df[column_name].map(map)
    return df

def filter_lines(df, column_name, value):
        """
        Filters the DataFrame `df` based on the given `column_name` and `value`.

        Parameters:
        - df (pandas.DataFrame): The DataFrame to be filtered.
        - column_name (str): The name of the column to filter on.
        - value: The value to filter for in the specified column.

        Returns:
        - pandas.DataFrame: The filtered DataFrame.

        Example:
        >>> df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]})
        >>> filtered_df = filter_lines(df, 'Name', 'Bob')
        >>> print(filtered_df)
            Name  Age
        1   Bob   30
        """
        return df[df[column_name] == value]

def decoder(df):
    """
    Decodes the categorical variables in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the categorical variables to be decoded.

    Returns:
        pandas.DataFrame: The DataFrame with the categorical variables decoded.
    """
    df['gender'] = df['gender'].map(reverse_gender_mapping)
    df['disability'] = df['disability'].map(reverse_disability_mapping)
    df['final_result'] = df['final_result'].map(reverse_final_result_mapping)
    df['age_band'] = df['age_band'].map(reverse_age_mapping)
    df['highest_education'] = df['highest_education'].map(reverse_highest_education_mapping)
    df['imd_band'] = df['imd_band'].map(reverse_imd_band_mapping)
    df['code_presentation'] = df['code_presentation'].map(reverse_code_presentation_mapping)
    df['code_module'] = df['code_module'].map(reverse_code_module_mapping)
    df['region'] = df['region'].map(reverse_region_mapping)

    return df

def encoder(df):
    """
    Encodes categorical variables in the given DataFrame using predefined mapping dictionaries.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the categorical variables to be encoded.

    Returns:
    pandas.DataFrame: The DataFrame with the categorical variables encoded.
    """
    df['gender'] = df['gender'].map(gender_mapping)
    df['disability'] = df['disability'].map(disability_mapping)
    df['final_result'] = df['final_result'].map(final_result_mapping)
    df['age_band'] = df['age_band'].map(age_mapping)
    df['highest_education'] = df['highest_education'].map(highest_education_mapping)
    df['imd_band'] = df['imd_band'].map(imd_band_mapping)
    df['code_presentation'] = df['code_presentation'].map(code_presentation_mapping)
    df['code_module'] = df['code_module'].map(code_module_mapping)
    df['region'] = df['region'].map(region_mapping)
    
    return df
    
def is_clean(df):
    """
    Checks if a DataFrame is clean.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be checked.

    Returns:
    bool: True if the DataFrame is clean (no NaN values and no object dtype columns), False otherwise.
    """
    # Check for NaN values
    if df.isnull().sum().sum() > 0 or 'object' in df.dtypes.values:
        return False
    else:
        return True