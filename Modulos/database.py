import json
import pickle
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import pandas as pd
from dotenv import load_dotenv
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cleaning
import bcrypt

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
            query = text("INSERT INTO dataframes (df_name, df_type) VALUES (:df_name, :df_type) RETURNING df_id")
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

def retrieve_dataset(database_name, df_id):
    """
    retrieves a dataset from a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - df_id (int): The ID of the dataset to retrieve.

    Returns:
    - df (pandas.DataFrame): The retrieved dataset as a pandas DataFrame.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
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

        # retrieve the dataset
        df = pd.read_sql(query, engine)
        return df
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the dataset {df_id} from data table in {database_name}.")
        print(str(e))

def list_datasets(database_name):
    """
    Lists all datasets stored in the PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.

    Returns:
    - dataset_list (pandas.DataFrame): the table with the list of datasets.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
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

def store_model(model, database_name, model_name, model_type):
    """
    Store a model in a PostgreSQL database using sqlalchemy.

    Parameters:
    - model (object): The model to store.
    - database_name (str): The name of the database.
    - model_name (str): The name of the model.
    - model_type (int): The type of the model. (1=SVM, 2=XGBOOST, 3=Random Forest)
    """
    params = json.dumps(model.get_params())

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Store the model in the database
        with engine.connect() as connection:

            # Insert a record into the models table
            query = text(f"""
                         INSERT INTO models (
                         model_name, model_type, model_file_name, parameters) 
                         VALUES (:model_name, :model_type, :model_file_name, :parameters) 
                         RETURNING model_id
                         """)
            
            result = connection.execute(query, \
                                        model_name=model_name, \
                                            model_type=model_type, \
                                                model_file_name=model_name + '.pkl', \
                                                    parameters=params)

    except SQLAlchemyError as e:
        print(f"An error occurred while storing the model {model_name} in {database_name}.")
        print(str(e))

    # Save the model to a file
    save_model_file(model, model_name)

def save_model_file(model, model_name):
    """
    Save a model to a file in the model folder.

    Parameters:
    - model (object): The model to save.
    - model_name (str): The name of the model.
    """
    load_dotenv()
    folder_path = os.getenv('MODEL_FILE_PATH')

    # Combine the folder path and the filename
    file_path = os.path.join(folder_path, model_name + '.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def retrieve_model(database_name, model_id):
    """
    retrieves a model from a PostgreSQL database and saved file.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - model_id (int): The ID of the model to retrieve.

    Returns:
    - model (object): The model to retrieve.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Define the SQL query to select rows from the model table
        query = f"""
        SELECT *
        FROM models
        WHERE model_id = {model_id}
        """

        # retrieve the dataframe with the model info
        model_info = pd.read_sql(query, engine)

        # Get the model file name from the dataframe
        model_file_name = model_info['model_file_name'].values[0]

        # Combine the folder path and the model file name
        load_dotenv()
        folder_path = os.getenv('MODEL_FILE_PATH')
        model_file_path = os.path.join(folder_path, model_file_name)

        # Load the model from the file
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)

        return model
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the model info {model_id} from model table in {database_name}.")
        print(str(e))

def list_models(database_name):
    """
    Lists all models stored in the PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.

    Returns:
    - model_list (pandas.DataFrame): the table with the list of models.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Define the SQL query to select all records from the dataFrames table
        query = "SELECT * FROM models"

        # Execute the query and return the result as a DataFrame
        model_list = pd.read_sql(query, engine)
        return model_list
    
    except SQLAlchemyError as e:
        print(f"An error occurred while listing the models from {database_name}.")
        print(str(e))

""" # Not needed, stored like a dataset
def store_prediction(df, database_name, dataset_name):
    pass

def retrieve_prediction(database_name, dataset_name):
    pass
 """

def update_df_history(database_name, df_id, change):
    """
    Store a change in a dataframe in the dataframe_changes table.
    
    Parameters:
    - database_name (str): The name of the database.
    - df_id (int): The ID of the dataframe.
    - change (str): The description of the change.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Store the model in the database
        with engine.connect() as connection:

            # Insert a record into the models table
            query = text(f"""
                         INSERT INTO dataframe_changes (
                         df_id, change_description) 
                         VALUES (:df_id, :change_description) 
                         RETURNING change_id
                         """)
            
            result = connection.execute(query, \
                                        df_id=df_id, \
                                            change_description=change)

    except SQLAlchemyError as e:
        print(f"An error occurred while storing the dataframe history update in {database_name}.")
        print(str(e))

def retrieve_df_history(database_name, df_id):
    """
    retrieves the changes made to a dataframe from a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - df_id (int): The ID of the df whose changes are to be retrieved.

    Returns:
    - history (pandas.Dataframe): The dataframe containing all changes to the dataframe.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Define the SQL query to select rows from the model table
        query = f"""
        SELECT *
        FROM dataframe_changes
        WHERE df_id = {df_id}
        """

        # retrieve the dataframe with the model info
        history = pd.read_sql(query, engine)

        return history
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the dataframe history {df_id} from dataframe_changes table in {database_name}.")
        print(str(e))

def store_evaluations(database_name, model_id, fp, fn, tp, tn):
    """
    Store the evaluations of a model in the evaluations table.
    
    Parameters:
    - database_name (str): The name of the database.
    - model_id (int): The ID of the model.
    - fp (int): The number of false positives.
    - fn (int): The number of false negatives.
    - tp (int): The number of true positives.
    - tn (int): The number of true negatives.
    """
    """
    create table evaluations (
    evaluation_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(model_id),
    fp INTEGER,
    fn INTEGER,
    tp INTEGER,
    tn INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Store the evaluation in the database
        with engine.connect() as connection:

            # Insert a record into the models table
            query = text(f"""
                         INSERT INTO evaluations (
                         model_id, fp, fn, tp, tn) 
                         VALUES (:model_id, :fp, :fn, :tp, :tn) 
                         RETURNING evaluation_id
                         """)
            
            result = connection.execute(query, \
                                        model_id=model_id, \
                                            fp=fp, \
                                                fn=fn, \
                                                    tp=tp, \
                                                        tn=tn)

    except SQLAlchemyError as e:
        print(f"An error occurred while storing the model {model_id} eval in {database_name}.")
        print(str(e))


def retrieve_evaluations(database_name, model_id):
    """
    retrieves the evaluations of a model from a PostgreSQL database.
    
    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - model_id (int): The ID of the model whose evaluations are to be retrieved.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Define the SQL query to select rows from the model table
        query = f"""
        SELECT *
        FROM evaluations
        WHERE model_id = {model_id}
        """

        # retrieve the dataframe with the model info
        evaluations = pd.read_sql(query, engine)

        return evaluations
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the model {model_id} eval history in {database_name}.")
        print(str(e))

def retrieve_parameters(database_name, model_id):
    """
    retrieves the parameters of a model from a PostgreSQL database.
    
    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - model_id (int): The ID of the model whose parameters are to be retrieved.

    Returns:
    - parameters (JSON): JSON containing the parameters of the model.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Define the SQL query to select rows from the model table
        query = f"""
        SELECT parameters
        FROM models
        WHERE model_id = {model_id}
        """

        # Execute the SQL query and fetch the result
        result = engine.execute(query)

        # Retrieve the parameters as a JSON
        parameters = json.loads(result.fetchone()[0])

        return parameters
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the model {model_id} parameters in {database_name}.")
        print(str(e))

def create_user(full_name, email, num_id, type, database_name):
    """
    Create a new user in the users table.
    
    Parameters:
    - full_name (str): The name of the user.
    - email (str): The email of the user.
    - num_id (int): The work ID of the user.
    - type (int): The type of user account. (1=Admin, 2=teacher, 3=DataScientist)
    - database_name (str): The name of the database.
    """
    # Hash the default password
    hashed_default_password = bcrypt.hashpw("1234".encode(), bcrypt.gensalt())

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Store the user in the database
        with engine.connect() as connection:

            # Insert a record into the users table
            query = text(f"""
                         INSERT INTO users (
                         full_name, email, password, num_id, type) 
                         VALUES (:full_name, :email, :num_id, :type) 
                         RETURNING user_id
                         """)
            
            result = connection.execute(query, \
                                        full_name=full_name, \
                                                email=email, \
                                                    password=hashed_default_password, \
                                                    num_id=num_id, \
                                                        type=type)
            
            user_id = result.fetchone()[0]

            if type == 1:
                # Insert a record into the admins table
                query = text(f"""
                             INSERT INTO administrators (
                             admin_id) 
                             VALUES (:user_id)
                             """)
                connection.execute(query, user_id=user_id)
            elif type == 2:
                # Insert a record into the teachers table
                query = text(f"""
                             INSERT INTO teachers (
                             teacher_id) 
                             VALUES (:user_id)
                             """)
                connection.execute(query, user_id=user_id)
            else:
                # Insert a record into the scientist table
                query = text(f"""
                             INSERT INTO scientists (
                             scientist_id) 
                             VALUES (:user_id)
                             """)
                connection.execute(query, user_id=user_id)
            

    except SQLAlchemyError as e:
        print(f"An error occurred while storing the user {email} in {database_name}.")
        print(str(e))
        

def list_users(database_name):
    """
    Lists all users stored in the PostgreSQL database.
    
    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    
    Returns:
    - user_list (pandas.DataFrame): the table with the list of users.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Define the SQL query to select all records from the users table
        query = """SELECT 
                user_id, 
                full_name, 
                email, 
                num_id, 
                type, 
                is_active,
                created_at
                FROM users"""

        # Execute the query and return the result as a DataFrame
        user_list = pd.read_sql(query, engine)
        return user_list
    
    except SQLAlchemyError as e:
        print(f"An error occurred while listing the users from {database_name}.")
        print(str(e))

def update_user(database_name, email, column_to_update, new_value):
    """
    Update a user in the users table.
    
    Parameters:
    - database_name (str): The name of the database.
    - email (str): The email of the user to update.
    - column_to_update (str): The name of the column to update.
    - new_value: The new value of the user.
    """

    if column_to_update in ['full_name', 'email', 'num_id']:
        try:
            # Create a connection to PostgreSQL database
            engine = create_engine(database_name)

            # Store the user in the database
            with engine.connect() as connection:

                # Update the user in the users table
                query = text(f"""
                            UPDATE users
                            SET 
                            {column_to_update} = :new_value
                            WHERE email = :email
                            """)
                
                connection.execute(query, \
                                            new_value=new_value, \
                                                    email=email)
                

        except SQLAlchemyError as e:
            print(f"An error occurred while updating the user {email} in {database_name}.")
            print(str(e))
    else:
        print(f"Column {column_to_update} not valid.")

def change_password(database_name, email, new_password, old_password):
    """
    Change the password of a user in the users table.
    
    Parameters:
    - database_name (str): The name of the database.
    - email (str): The email of the user to update.
    - new_password: The new password of the user.
    - old_password: The old password of the user to confirm security.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Store the user in the database
        with engine.connect() as connection:

            # Get the hashed password from the database
            result = connection.execute("SELECT password FROM users WHERE email = :email", email=email)
            db_password = result.fetchone()[0]

            # check if old password is correct
            if bcrypt.checkpw(old_password.encode(), db_password):

                # Hash the new password
                new_password_hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())

                # Update the user in the users table
                query = text("""
                            UPDATE users
                            SET 
                            password = :new_password
                            WHERE email = :email
                            """)

                connection.execute(query, new_password=new_password_hashed, email=email)
            else:
                print("The old password is incorrect.")
    except Exception as e:
        print(f"An error occurred: {e}")

def deactivate_user(database_name, email):
    """
    Deactivate a user in the users table.
    
    Parameters:
    - database_name (str): The name of the database.
    - email (str): The email of the user to deactivate.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Store the user in the database
        with engine.connect() as connection:

            # Update the user in the users table
            query = text("""
                        UPDATE users
                        SET 
                        is_active = FALSE
                        WHERE email = :email
                        """)

            connection.execute(query, email=email)

    except SQLAlchemyError as e:
        print(f"An error occurred while deactivating the user {email} in {database_name}.")
        print(str(e))

def is_password_correct(database_name, email, password):
    """
    Check if the password of a user is correct.
    
    Parameters:
    - database_name (str): The name of the database.
    - email (str): The email of the user.
    - password: The password provided by the user.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(database_name)

        # Store the user in the database
        with engine.connect() as connection:

            # Get the hashed password from the database
            result = connection.execute("SELECT password FROM users WHERE email = :email", email=email)
            db_password = result.fetchone()[0]

            # check if password is correct
            if bcrypt.checkpw(password.encode(), db_password):
                return True
            else:
                return False

    except SQLAlchemyError as e:
        print(f"An error occurred while checking password of user {email} in {database_name}.")
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

def store_prediction(df, database_name, dataset_name):
    pass