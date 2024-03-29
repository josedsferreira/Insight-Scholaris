import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import json
from dotenv import load_dotenv

load_dotenv()
column_names = os.getenv('COLUMN_NAMES').split(',')

def clean_data(df):
    """
    Cleans the given DataFrame by applying necessary transformations.

    Args:
        df (pandas.DataFrame): The DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    if is_df_clean(df):
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

    # Open the JSON file
    with open('decoding.json', 'r') as f:
        # Load the JSON file into a dictionary
        mappings = json.load(f)

    # Replace values in each column based on the mappings
    for column, mapping in mappings.items():
        if column in df.columns:
            # Convert keys in mapping to int, because JSON keys are always strings
            mapping = {int(k): v for k, v in mapping.items()}
            df[column] = df[column].map(mapping)

    return df

def encoder(df):
    """
    Encodes categorical variables in the given DataFrame using predefined mapping dictionaries.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the categorical variables to be encoded.

    Returns:
    pandas.DataFrame: The DataFrame with the categorical variables encoded.
    """

    # Open the JSON file
    with open('encoding.json', 'r') as f:
        # Load the JSON file into a dictionary
        mappings = json.load(f)

    # Replace values in each column based on the mappings
    for column, mapping in mappings.items():
        if column in df.columns:
            # Convert keys in mapping to int, because JSON keys are always strings
            mapping = {int(k): v for k, v in mapping.items()}
            df[column] = df[column].map(mapping)

    return df
    
def is_df_clean(df):
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
    
def is_df_encoded(df):
    """
    Checks if a DataFrame is encoded.

    Args:
        df (pandas.DataFrame): The DataFrame to be checked.

    Returns:
        bool: True if the DataFrame is encoded, False otherwise.
    """
    # Open the JSON file
    with open('encoding.json', 'r') as f:
        # Load the JSON file into a dictionary
        mappings = json.load(f)

    for column, mapping in mappings.items():
        if column in df.columns:
            # Check if all values in the column are in the mapping
            if not set(df[column].unique()).issubset(set(mapping.values())):
                return False

    return True
    