from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from dotenv import load_dotenv
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def create_engine(database_name):
    """
    Create and return a SQLAlchemy engine for connecting to a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the database to connect to.

    Returns:
    - engine (sqlalchemy.engine.Engine): The SQLAlchemy engine object.

    """
    # Get user and password from .env
    load_dotenv()
    user = os.getenv('USER_POSTGRES')
    password = os.getenv('PASSWORD_POSTGRES')
    connection_link = "postgresql://" + user + ":" + password + "@localhost:5432/" + database_name
    # Create a connection to PostgreSQL database
    engine = create_engine(connection_link)
    return engine

def store_dataset(df, database_name, dataset_name):
    """
    Store a dataframe in a PostgreSQL database using sqlalchemy
    df: DataFrame
    database_name: Name of the database
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)
        # Store the dataframe in the database
        df.to_sql(dataset_name, engine, if_exists='replace', index=False)
    except SQLAlchemyError as e:
        print(f"An error occurred while storing the dataset {dataset_name} in {database_name}.")
        print(str(e))

    """
    other possible arguments for .to_sql:
    schema: Name of the schema
    dtype: Dictionary with the type of each column, if not set, it will be inferred
    """

def download_dataset(database_name, dataset_name):
    """
    Downloads a dataset from a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - dataset_name (str): The name of the dataset to download.

    Returns:
    - df (pandas.DataFrame): The downloaded dataset as a pandas DataFrame.

    Raises:
    - SQLAlchemyError: If an error occurs while downloading the dataset.

    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)
        # Download the dataset
        df = pd.read_sql(dataset_name, engine)
        return df
    except SQLAlchemyError as e:
        print(f"An error occurred while downloading the dataset {dataset_name} from {database_name}.")
        print(str(e))

def choose_file():
    """
    Opens a file picker dialog and returns the path of the selected file.

    Returns:
        str: The path of the selected file.
    """
    # Hide the main tkinter window
    Tk().withdraw()

    # Open the file picker dialog and get the path of the selected file
    filename = askopenfilename()

    # Return the path of the selected file
    return filename

def open_dataframe_from_file(path):
    """
    Opens a dataframe from a file.

    Args:
        path (str): The path to the file.

    Returns:
        pandas.DataFrame or None: The dataframe read from the file, or None if an error occurred.
    """
    root_path, file_extension = os.path.splitext(path)
    try:
        if file_extension == '.csv':
            df = pd.read_csv(path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return None
        return df
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"An error occurred while opening the file: {str(e)}")

def obtain_dataset_list(database_name):
    """
    Obtain the list of datasets from the specified PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.

    Returns:
    - datasets (list): A list of dataset names in the database.

    Raises:
    - SQLAlchemyError: If an error occurs while obtaining the list of datasets.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)
        # Obtain the list of datasets
        datasets = engine.table_names()
        return datasets
    except SQLAlchemyError as e:
        print(f"An error occurred while obtaining the list of datasets from {database_name}.")
        print(str(e))

def store_prediction(df, database_name, dataset_name):
    pass