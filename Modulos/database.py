from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import pandas as pd
from dotenv import load_dotenv
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cleaning

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

def store_dataset(df, db_name, df_name, df_type):
    """
    Store a dataframe in a PostgreSQL database using sqlalchemy.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame to store.
    - db_name (str): The name of the database.
    - df_name (str): The name of the DataFrame.
    - df_type (str): The type of the DataFrame.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(db_name)
        
        #encode the dataframe if it is not encoded
        if not cleaning.is_df_encoded(df):
            print("The DataFrame is not encoded. Encoding it now...")
            df = cleaning.encoder(df)

        # Store the dataframe in the database
        with engine.connect() as connection:
            # Insert a record into the dataFrames table
            query = text("INSERT INTO dataFrames (df_name, df_type) VALUES (:df_name, :df_type) RETURNING df_id")
            result = connection.execute(query, df_name=df_name, df_type=df_type)
            df_id = result.fetchone()[0]

            # Add df_id column to the DataFrame
            df['df_id'] = df_id

            # Store the DataFrame in the data table
            df.to_sql('data', engine, if_exists='append', index=False)

    except SQLAlchemyError as e:
        print(f"An error occurred while storing the dataset {df_name} in {db_name}.")
        print(str(e))

    """
    other possible arguments for .to_sql:
    schema: Name of the schema
    dtype: Dictionary with the type of each column, if not set, it will be inferred
    """

def download_dataset(database_name, df_id):
    """
    Downloads a dataset from a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - df_id (int): The ID of the dataset to download.

    Returns:
    - df (pandas.DataFrame): The downloaded dataset as a pandas DataFrame.

    Raises:
    - SQLAlchemyError: If an error occurs while downloading the dataset.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Define the SQL query to select rows from the data table
        query = f"""
        SELECT *
        FROM data
        WHERE df_id = {df_id}
        """

        # Download the dataset
        df = pd.read_sql(query, engine)
        return df
    
    except SQLAlchemyError as e:
        print(f"An error occurred while downloading the dataset {df_id} from data table in {database_name}.")
        print(str(e))

def list_datasets(database_name):
    """
    Lists all datasets stored in the PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.

    Returns:
    - dataset_list (pandas.DataFrame): the table with the list of datasets.

    Raises:
    - SQLAlchemyError: If an error occurs while downloading the dataset.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Define the SQL query to select all records from the dataFrames table
        query = "SELECT * FROM dataFrames"

        # Execute the query and return the result as a DataFrame
        dataset_list = pd.read_sql(query, engine)
        return dataset_list
    
    except SQLAlchemyError as e:
        print(f"An error occurred while listing the datasets from {database_name}.")
        print(str(e))

def store_model(model, database_name, model_name):
    pass

def download_model(database_name, model_name):
    pass

def store_prediction(df, database_name, dataset_name):
    pass

def download_prediction(database_name, dataset_name):
    pass

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

def store_prediction(df, database_name, dataset_name):
    pass