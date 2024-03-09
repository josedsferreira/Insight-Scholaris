import pandas as pd
from sklearn.preprocessing import LabelEncoder

# List of column names in datasets
column_names = ["code_module","code_presentation","id_student","gender","region",\
                "highest_education","imd_band","age_band","num_of_prev_attempts",\
                    "studied_credits","disability","final_result"]

# Define the mapping dictionaries
gender_mapping = {'M': 1, 'F': 2, 'Other': 3, 'Unknown': 0}

disability_mapping = {'N': 1, 'Y': 2, 'Unknown': 0}

final_result_mapping = {'Pass': 1, 'Distinction': 1, 'Withdrawn': 2, 'Fail': 2, 'Unknown': 0}

age_mapping = {'0-35': 1, '35-55': 2, '55<=': 3, 'Unknown': 0}

highest_education_mapping = {'No Formal quals': 1, \
                             'Lower Than A Level': 2, \
                                'A Level or Equivalent': 3, \
                                    'HE Qualification': 4, \
                                        'Post Graduate Qualification': 5, \
                                            'Unknown': 0}

imd_band_mapping = {'0-10%': 1, '10-20': 2, '20-30%': 3, '30-40%': 4, '40-50%': 5, \
                    '50-60%': 6, '60-70%': 7, '70-80%': 8, '80-90%': 9, '90-100%': 10, \
                        'Unknown': 0}

code_presentation_mapping = {'2013B': 1, '2013J': 2, '2014B': 3, '2014J': 4, 'Unknown': 0}

code_module_mapping = {'AAA': 1, 'BBB': 2, 'CCC': 3, 'DDD': 4, \
                       'EEE': 5, 'FFF': 6, 'GGG': 7, 'Unknown': 0}

region_mapping = {'Scotland': 1, \
                  'East Anglian Region': 2, \
                    'London Region': 3, \
                        'South Region': 4, \
                            'North Western Region': 5, \
                                'West Midlands Region': 6, \
                                    'South West Region': 7, \
                                        'East Midlands Region': 8, \
                                            'Wales': 9, \
                                                'Yorkshire Region': 10, \
                                                    'North Region': 11, \
                                                        'Ireland': 12, \
                                                            'South East Region': 13, \
                                                                'Central Region': 14, \
                                                                    'North Western Region': 15, \
                                                                        'Unknown': 0}

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
        df = nans_to_unknown(df)
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

def remove_nans_columns(df, column_list=column_names):
    """
    Remove rows with NaN values in the specified columns from the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to remove rows from.
    column_list (list): A list of column names to check for NaN values.

    Returns:
    pandas.DataFrame: The DataFrame with rows containing NaN values in the specified columns removed.
    """
    return df.dropna(subset=column_list)

def average_nans(df, column_list=column_names):
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

def mode_nans(df, column_list=column_names):
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

def nans_to_unknown(df, column_list=column_names):
    """
    Fill missing values in the specified columns of a DataFrame with the string 'Unknown'.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the columns with missing values.
    - column_list (list): A list of column names to fill the missing values.

    Returns:
    - df (pandas.DataFrame): The DataFrame with missing values filled with the string 'Unknown'.
    """
    for column_name in column_list:
        df[column_name].fillna(0, inplace=True)
    return df

def remove_columns(df, column_list=column_names):
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