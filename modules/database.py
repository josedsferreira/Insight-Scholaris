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
from modules import data
import bcrypt

def connection_link(database_name):
    """
    Create a connection link to a PostgreSQL database.
    
    Parameters:
    - database_name (str): The name of the database.
    
    Returns:
    - connection_link (str): The connection link to the database.
    """

    # Get user and password from .env
    load_dotenv()
    user = os.getenv('USER_POSTGRES')
    password = os.getenv('PASSWORD_POSTGRES')
    connection_link = "postgresql://" + user + ":" + password + "@localhost:5432/" + database_name
    return connection_link

def store_dataset(df, db_name, df_name, df_type):
    """
    Store a dataframe in a PostgreSQL database using sqlalchemy.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame to store.
    - db_name (str): The name of the database.
    - df_name (str): The name of the DataFrame.
    - df_type (str): The type of the DataFrame.

    Returns:
    - success (bool): True if the DataFrame was stored successfully, False otherwise.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(db_name))

        # Drop all rows from df that don't have a value for 'final_result' or that have NaN
        df = df.dropna(subset=['final_result'])
        
        #encode the dataframe if it is not encoded
        if not data.is_df_encoded(df):
            print("The DataFrame is not encoded. Encoding it now...")
            df = data.encoder(df)

        # Store the dataframe in the database
        with engine.connect() as connection:

            # Insert a record into the dataFrames table
            query = text("""
                         INSERT INTO dataframes (df_name, df_type, num_cols, num_rows, num_unknowns, num_missing) 
                         VALUES (:df_name, :df_type, :num_cols, :num_rows, :num_unknowns, :num_missing) 
                         RETURNING df_id
                         """)

            # create dictionary with dataframe info for df_info
            info_dict = data.create_dataframe_info(df)

            params = {'df_name': df_name, 'df_type': df_type, 'num_cols': info_dict['num_columns'], 'num_rows': info_dict['num_rows'], 'num_unknowns': info_dict['unknowns'], 'num_missing': info_dict['missing_values']}
            
            result = connection.execute(query, params)
            connection.commit()
            df_id = result.fetchone()[0]

            # Add df_id column to the DataFrame
            df['dataframe_id'] = df_id

            # Store the DataFrame in the data table
            df.to_sql('data', engine, if_exists='append', index=False)

            return True
    except SQLAlchemyError as e:
        print(f"An error occurred while storing the dataset {df_name} in {db_name}.")
        print(str(e))
        return False

    """
    other possible arguments for .to_sql:
    schema: Name of the schema
    dtype: Dictionary with the type of each column, if not set, it will be inferred
    """

def retrieve_dataset_info(database_name, df_id):
    """
    retrieves a dataset from a PostgreSQL database. Attention, JSON variables
    become dictionaries when retrieved.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - df_id (int): The ID of the dataset to retrieve.

    Returns:
    - df_info (pandas.DataFrame): The retrieved dataset as a pandas DataFrame.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))
        with engine.connect() as connection:

            query = text("""
                    SELECT 
                    df_id AS "ID",
                    df_name AS "Nome",
                    CASE
                        WHEN df_type = 1 THEN 'De treino'
                        WHEN df_type = 2 THEN 'Para prever'
                        WHEN df_type = 3 THEN 'Previsão'
                    END AS "Tipo",
                    num_cols AS "Numero de colunas",
                    num_rows AS "Numero de linhas",
                    num_unknowns AS "Numero de desconhecidos",
                    num_missing AS "Numero de valores em falta",
                    date_trunc('day', created_at) AS "Criado em"
                    FROM dataFrames 
                    WHERE df_id = :id
                    ORDER BY "Criado em" DESC
                    """)

            result = connection.execute(query, {'id': df_id})
            df_info = result.fetchone()
            return df_info
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the dataset {df_id} from dataFrames table in {database_name}.")
        print(str(e))

def retrieve_dataset(database_name, df_id, decode=True):
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
        engine = create_engine(connection_link(database_name))

        df_info = retrieve_dataset_info(database_name, df_id)
        df_type = df_info[2]

        # Define the SQL query to select rows from the data table
        if df_type == 'Para prever':
            query = text("""
                     SELECT
                     code_module,
                     code_presentation,
                     gender,
                     region,
                     highest_education,
                     imd_band,
                     age_band,
                     num_of_prev_attempts,
                     studied_credits,
                     disability
                     FROM data
                     WHERE dataframe_id = :df_id
                     """)
        else:
            query = text("""
                        SELECT
                        code_module,
                        code_presentation,
                        gender,
                        region,
                        highest_education,
                        imd_band,
                        age_band,
                        num_of_prev_attempts,
                        studied_credits,
                        disability,
                        final_result
                        FROM data
                        WHERE dataframe_id = :df_id
                        """)

        params = {'df_id': df_id}
        
        df = pd.read_sql(query, engine, params=params)

        if decode:
            # Decode the DataFrame
            df = data.decoder(df)

        return df
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the dataset {df_id} from data table in {database_name}.")
        print(str(e))

def retireve_head(database_name, df_id, n_rows):
    """
    retrieves the first n rows of a dataset from a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - df_id (int): The ID of the dataset to retrieve.
    - n_rows (int): The number of rows to retrieve.

    Returns:
    - df (pandas.DataFrame): The retrieved dataset as a pandas DataFrame.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
    """
    try:
        engine = create_engine(connection_link(database_name))

        df_type = retrieve_dataset_info(database_name, df_id)[2]

        # Define the SQL query to select rows from the data table
        if df_type == 'Para prever':
            query = text("""
                     SELECT
                     code_module,
                     code_presentation,
                     gender,
                     region,
                     highest_education,
                     imd_band,
                     age_band,
                     num_of_prev_attempts,
                     studied_credits,
                     disability
                     FROM data
                     WHERE dataframe_id = :df_id
                     LIMIT :n_rows
                     """)
        else:
            query = text("""
                        SELECT
                        code_module,
                        code_presentation,
                        gender,
                        region,
                        highest_education,
                        imd_band,
                        age_band,
                        num_of_prev_attempts,
                        studied_credits,
                        disability,
                        final_result
                        FROM data
                        WHERE dataframe_id = :df_id
                        LIMIT :n_rows
                        """)

        params = {'df_id': df_id, 'n_rows': n_rows}
        
        head = pd.read_sql(query, engine, params=params)
        head = data.decoder(head)
        return head
    
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
        engine = create_engine(connection_link(database_name))

        # Define the SQL query to select all records from the dataFrames table
        query = """ 
                SELECT 
                df_id AS "ID",
                df_name AS "Nome",
                CASE
                    WHEN df_type = 1 THEN 'De treino'
                    WHEN df_type = 2 THEN 'Para prever'
                    WHEN df_type = 3 THEN 'Previsão'
                END AS "Tipo",
                date_trunc('day', created_at) AS "Criado em"
                FROM dataFrames 
                ORDER BY "Criado em" DESC
                """

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

    Returns:
    - model_id (int): The ID of the stored model.
    """

    model_params = model.get_params()
    # Convert 'Missing' value from NaN to None so that JSON accepts it
    model_params['missing'] = None
    json_params = json.dumps(model_params)

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the model in the database
        with engine.connect() as connection:

            # Insert a record into the models table
            query = text(f"""
                         INSERT INTO models (
                         model_name, model_type, parameters) 
                         VALUES (:model_name, :model_type, :parameters) 
                         RETURNING model_id
                         """)
            
            params = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'parameters': json_params
                    }
            
            result = connection.execute(query, params)
            model_id = result.fetchone()[0]
            connection.commit()

            file_name = set_model_file_name(database_name, model_id, model_name)

            # Save the model to a file
            save_model_file(model, file_name)
            print(f"Model {model_name} stored successfully in {database_name}. ID: {model_id}")
            return model_id

    except SQLAlchemyError as e:
        print(f"An error occurred while storing the model {model_name} in {database_name}.")
        print(str(e))
        return None
    
def set_model_file_name(database_name, model_id, model_name):
    """
    Set the model file name in the models table.
    
    Parameters:
    - database_name (str): The name of the database.
    - model_id (int): The ID of the model.
    - model_name (str): The name of the model file.

    Returns:
    - file_name (str): The name of the model file.
    """

    file_name = model_name + '_' + str(model_id) + '.pkl'

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the model in the database
        with engine.connect() as connection:

            # Update the model in the models table
            query = text(f"""
                        UPDATE models
                        SET 
                        model_file_name = :file_name
                        WHERE model_id = :model_id
                        """)
            
            params = {'file_name': file_name, 'model_id': model_id}
            
            connection.execute(query, params)
            connection.commit()
            return file_name

    except SQLAlchemyError as e:
        print(f"An error occurred while updating the model file name for id {model_id} in {database_name}.")
        print(str(e))

def update_model_file(database_name, model_id, model):
    """
    Update a model file using info from the database.
    
    Parameters:
    - database_name (str): The name of the database.
    - model_id (int): The ID of the model to update.

    Returns:
    - success (bool): True if the model was updated successfully, False otherwise.
    """

    try:
        # first retrieve the model info
        model_info = retrieve_model_info(database_name, model_id)
        model_name = model_info['model_name'].values[0]
        save_model_file(model, model_name)
        return True
    except SQLAlchemyError as e:
        print(f"An error occurred while updating the model file for id {model_id} .")
        print(str(e))
        return False

def retrieve_model_info(database_name, model_id):
    """
    retrieves a model from a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - model_id (int): The ID of the model to retrieve.

    Returns:
    - model_info (pandas.DataFrame): The dataframe containing the model info.

    Raises:
    - SQLAlchemyError: If an error occurs while retrieving the dataset.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Define the SQL query to select rows from the model table
        query = text("""
        SELECT *
        FROM models
        WHERE model_id = :model_id
        """)

        params = {'model_id': int(model_id)}

        # retrieve the dataframe with the model info
        model_info = pd.read_sql(query, engine, params=params)

        return model_info
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the model info {model_id} from model table in {database_name}.")
        print(str(e))

def save_model_file(model, file_name):
    """
    Save a model to a file in the model folder.

    Parameters:
    - model (object): The model to save.
    - file_name (str): The name to be given to the file.
    """
    load_dotenv()
    folder_path = os.getenv('MODEL_FOLDER_PATH')

    # Combine the folder path and the filename
    file_path = os.path.join(folder_path, file_name)

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
        engine = create_engine(connection_link(database_name))

        # Define the SQL query to select rows from the model table
        query = text("""
        SELECT *
        FROM models
        WHERE model_id = :model_id
        """)

        params={"model_id": model_id}

        # retrieve the dataframe with the model info
        model_info = pd.read_sql(query, engine, params=params)

        # Check if the model_id exists in the table
        if not model_info.empty:
            # Get the model file name from the dataframe
            model_file_name = model_info['model_file_name'].values[0]
        else:
            print(f"Model with id {model_id} not found.")
            model_file_name = None

        # Combine the folder path and the model file name
        load_dotenv()
        folder_path = os.getenv('MODEL_FOLDER_PATH')
        model_file_path = os.path.join(folder_path, model_file_name)

        # Load the model from the file
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)

        return model
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the model {model_id} from model table in {database_name}.")
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
        engine = create_engine(connection_link(database_name))

        # Define the SQL query to select all records from the dataFrames table
        query = "SELECT * FROM models"

        # Execute the query and return the result as a DataFrame
        model_list = pd.read_sql(query, engine)
        return model_list
    
    except SQLAlchemyError as e:
        print(f"An error occurred while listing the models from {database_name}.")
        print(str(e))

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
        engine = create_engine(connection_link(database_name))

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
        engine = create_engine(connection_link(database_name))

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

def store_evaluation(database_name, model_id, matrix):
    """
    Store the evaluations of a model in the evaluations table.
    
    Parameters:
    - database_name (str): The name of the database.
    - model_id (int): The ID of the model.
    - matrix (numpy.ndarray): The confusion matrix of the model.

    Returns:
    - success (bool): True if the evaluations were stored successfully, False otherwise.
    """

    fp, fn, tp, tn = matrix.ravel()
    
    # Convert numpy.int64 types to int
    fp, fn, tp, tn = int(fp), int(fn), int(tp), int(tn)
    
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the evaluation in the database
        with engine.connect() as connection:

            # Insert a record into the models table
            query = text(f"""
                         INSERT INTO evaluations (
                         model_id, fp, fn, tp, tn) 
                         VALUES (:model_id, :fp, :fn, :tp, :tn) 
                         RETURNING evaluation_id
                         """)
            
            params = {
                        'model_id': model_id,
                        'fp': fp,
                        'fn': fn,
                        'tp': tp,
                        'tn': tn
                    }
            
            result = connection.execute(query, params)
            connection.commit()

        return True
    except SQLAlchemyError as e:
        print(f"An error occurred while storing the model {model_id} eval in {database_name}.")
        print(str(e))
        return False

def retrieve_evaluations(database_name, model_id):
    """
    retrieves the evaluations of a model from a PostgreSQL database.
    
    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - model_id (int): The ID of the model whose evaluations are to be retrieved.

    Returns:
    - evaluations (pandas.DataFrame): The dataframe containing all evaluations of the model.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Define the SQL query to select rows from the model table
        query = text("""
        SELECT *
        FROM evaluations
        WHERE model_id = :model_id
        """)

        params = {'model_id': int(model_id)}

        # retrieve the dataframe with the model info
        evaluations = pd.read_sql(query, engine, params=params)

        return evaluations
    
    except SQLAlchemyError as e:
        print(f"An error occurred while retrieving the model {model_id} eval history in {database_name}.")
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

    Returns:
    - success (bool): True if the user was created successfully, False otherwise.
    """
    # Hash the default password
    hashed_default_password = bcrypt.hashpw("1234".encode(), bcrypt.gensalt())
    string_password = hashed_default_password.decode('utf-8')

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the user in the database
        with engine.connect() as connection:

            # Insert a record into the users table
            query = text(f"""
                         INSERT INTO users (
                         full_name, email, password, num_id, type) 
                         VALUES (:full_name, :email, :password, :num_id, :type) 
                         RETURNING user_id
                         """)
            
            params = {
                        'full_name': full_name,
                        'email': email,
                        'password': string_password,
                        'num_id': num_id,
                        'type': type
                    }
            
            result = connection.execute(query, params)
            connection.commit()
            
            if False: #tabelas de tipo de utilizador não necessarias
            
                user_id = result.fetchone()[0]
            
                if type == 1:
                    # Insert a record into the admins table
                    query = text(f"""
                                INSERT INTO administrators (
                                admin_id) 
                                VALUES (:user_id)
                                """)
                    params = {'user_id': user_id}
                    connection.execute(query, params)
                elif type == 2:
                    # Insert a record into the teachers table
                    query = text(f"""
                                INSERT INTO teachers (
                                teacher_id) 
                                VALUES (:user_id)
                                """)
                    params = {'user_id': user_id}
                    connection.execute(query, params)
                else:
                    # Insert a record into the scientist table
                    query = text(f"""
                                INSERT INTO scientists (
                                scientist_id) 
                                VALUES (:user_id)
                                """)
                    params = {'user_id': user_id}
                    connection.execute(query, params)
            
            return True #user created successfully
    except SQLAlchemyError as e:
        print(f"An error occurred while storing the user {email} in {database_name}.")
        print(str(e))
        return False #user not created
        
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
        engine = create_engine(connection_link(database_name))

        # Define the SQL query to select all records from the users table
        query = """
                SELECT 
                full_name AS "Nome", 
                email AS "E-mail", 
                num_id AS "ID", 
                CASE
                    WHEN type = 1 THEN 'Administrador'
                    WHEN type = 2 THEN 'Docente'
                    WHEN type = 3 THEN 'Cientista de Dados'
                END AS "Tipo",
                date_trunc('day', created_at) AS "Criado em"
                FROM users
                WHERE is_active = TRUE
                """

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
            engine = create_engine(connection_link(database_name))

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

def change_password(database_name, email, new_password):
    """
    Change the password of a user in the users table.
    
    Parameters:
    - database_name (str): The name of the database.
    - email (str): The email of the user to update.
    - new_password: The new password of the user.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the user in the database
        with engine.connect() as connection:

            # Hash the new password
            new_password_hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
            string_password = new_password_hashed.decode('utf-8')

            query = text("""
                         UPDATE users 
                         SET 
                         password = :new_password, 
                         default_pw = FALSE 
                         WHERE email = :email
                         """)
            
            params = {
                        'new_password': string_password,
                        'email': email
                    }

            # Get the hashed password from the database
            result = connection.execute(query, params)
            connection.commit()
            return True #password change successful

    except SQLAlchemyError as e:
        print(f"An error occurred while changing the password of user {email} in {database_name}.")
        print(str(e))
        return False #password change failed

def deactivate_user(database_name, email):
    """
    Deactivate a user in the users table.
    
    Parameters:
    - database_name (str): The name of the database.
    - email (str): The email of the user to deactivate.

    Returns:
    - success (bool): True if the user was deactivated successfully, False otherwise.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the user in the database
        with engine.connect() as connection:

            # Update the user in the users table
            query = text("""
                        UPDATE users
                        SET 
                        is_active = FALSE
                        WHERE email = :email
                        """)
            
            params = {'email': email}

            connection.execute(query, params)
            connection.commit()
            return True #user deactivated successfully

    except SQLAlchemyError as e:
        print(f"An error occurred while deactivating the user {email} in {database_name}.")
        print(str(e))
        return False #user not deactivated

def user_is_valid(database_name, email):
    """
    Check if a user exists in the users table.
    
    Parameters:
    - database_name (str): The name of the database.
    - email (str): The email of the user.
    
    Returns:
    - exists (bool): True if the user exists, False otherwise.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the user in the database
        with engine.connect() as connection:

            # Check if the user exists and is active
            sql = text("SELECT COUNT(*) FROM users WHERE email = :email and is_active = true")
            result = connection.execute(sql, {'email': email})
            count = result.fetchone()[0]

            if count > 0:
                return True
            else:
                return False

    except SQLAlchemyError as e:
        print(f"An error occurred while checking if user {email} exists in {database_name}.")
        print(str(e))

def user_info(database_name, email):
    """
    Get the type of a user in the users table.
    
    Parameters:
    - database_name (str): The name of the database.
    - email (str): The email of the user.
    
    Returns:
    - type (int): The type of the user.
    - default_pw (bool): True if the user is using the default password, False otherwise.
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the user in the database
        with engine.connect() as connection:

            # Get the type of the user and the default_pw value
            result = connection.execute(text("SELECT type, default_pw FROM users WHERE email = :email"), {'email': email})
            user_type, default_pw = result.fetchone()

            return user_type, default_pw

    except SQLAlchemyError as e:
        print(f"An error occurred while getting the type of user {email} in {database_name}.")
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
        engine = create_engine(connection_link(database_name))

        # Store the user in the database
        with engine.connect() as connection:

            # Get the hashed password from the database
            result = connection.execute(text("SELECT password FROM users WHERE email = :email"), {'email': email})
            db_password = result.fetchone()[0]

            # check if password is correct
            if bcrypt.checkpw(password.encode('utf-8'), db_password.encode('utf-8')):
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

def ready_export(database_name, df_id):
    """
    Prepare a dataset for export as a csv file that is saved to the static/downloads folder
    as download.csv

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - df_id (int): The ID of the dataset to export.

    Returns:
    - success (bool): True if the dataset was readied successfully, False otherwise.
    """
    try:
        engine = create_engine(connection_link(database_name))

        df_info = retrieve_dataset_info(database_name, df_id)
        df_type = df_info[2]

        # Define the SQL query to select rows from the data table
        if df_type == 'Para prever':
            query = text("""
                     SELECT
                     code_module,
                     code_presentation,
                     gender,
                     region,
                     highest_education,
                     imd_band,
                     age_band,
                     num_of_prev_attempts,
                     studied_credits,
                     disability
                     FROM data
                     WHERE dataframe_id = :df_id
                     """)
        else:
            query = text("""
                        SELECT
                        code_module,
                        code_presentation,
                        gender,
                        region,
                        highest_education,
                        imd_band,
                        age_band,
                        num_of_prev_attempts,
                        studied_credits,
                        disability,
                        final_result
                        FROM data
                        WHERE dataframe_id = :df_id
                        """)

        params = {'df_id': df_id}
        
        df = pd.read_sql(query, engine, params=params)

        # Decode the DataFrame
        df = data.decoder(df)

        # Export the DataFrame to a CSV file
        df.to_csv("static/downloads/data.csv", index=False)

        return True

    except SQLAlchemyError as e:
        print(f"An error occurred while exporting the dataset {df_id} from data table in {database_name}.")
        print(str(e))
        return False

def change_model_value(database_name, column_name, new_value, id):
    """
    Change the value of a column in the table 'models' in a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - column_name (str): The name of the column.
    - new_value: The new value of the column.
    - id: The ID of the row to update.

    Returns:
    - success (bool): True if the value was changed successfully, False otherwise.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the user in the database
        with engine.connect() as connection:

            # Update the value in the table
            query = text(f"""
                        UPDATE models
                        SET 
                        {column_name} = :new_value
                        WHERE model_id = :id
                        """)
            
            params = {'new_value': new_value, 'id': id}

            connection.execute(query, params)
            connection.commit()
            return True #value changed successfully

    except SQLAlchemyError as e:
        print(f"An error occurred while changing the value of {column_name} in model {id} in {database_name}.")
        print(str(e))
        return False #value not changed

def deactivate_all_models(database_name):
    """
    Deactivate all models in the 'models' table in a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.

    Returns:
    - success (bool): True if the models were deactivated successfully, False otherwise.
    """
    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Store the user in the database
        with engine.connect() as connection:

            # Update the value in the table
            query = text(f"""
                        UPDATE models
                        SET 
                        is_active = FALSE
                        """)
            
            connection.execute(query)
            connection.commit()
            return True #models deactivated successfully

    except SQLAlchemyError as e:
        print(f"An error occurred while deactivating all models in {database_name}.")
        print(str(e))
        return False #models not deactivated

def set_active_model(database_name, model_id):
    """
    Set the active model in the 'models' table in a PostgreSQL database.

    Parameters:
    - database_name (str): The name of the PostgreSQL database.
    - model_id (int): The ID of the active model.

    Returns:
    - success (bool): True if the active model was set successfully, False otherwise.
    """

    if deactivate_all_models(database_name):

        try:
            # Create a connection to PostgreSQL database
            engine = create_engine(connection_link(database_name))

            with engine.connect() as connection:

                # Update the value in the table
                query = text(f"""
                            UPDATE models
                            SET 
                            is_active = TRUE
                            WHERE model_id = :model_id
                            """)
                
                params = {'model_id': model_id}
    
                connection.execute(query, params)
                connection.commit()

                return True #active model set successfully

        except SQLAlchemyError as e:
            print(f"An error occurred while setting the active model in {database_name}.")
            print(str(e))
            return False #active model not set
    else:
        return False
    
def retrieve_active_model_info(database_name):
    """
    retrieves info about the active model, including evaluations from a PostgreSQL database.
    
    Parameters:
    - database_name (str): The name of the PostgreSQL database.

    Returns:
    - model_info (pandas.DataFrame): The dataframe containing all info of the active model.
    model_info has the following columns:

    0-'model_id'
    1-'model_name'
    2-'model_type'
    3-'is_trained'
    4-'is_active'
    5-'model_file_name'
    6-'parameters'
    7-'created_at'
    8-'evaluation_id'
    9-'fp'
    10-'fn'
    11-'tp'
    12-'tn'
    """

    try:
        # Create a connection to PostgreSQL database
        engine = create_engine(connection_link(database_name))

        # Define the SQL query to select rows from the model table
        query = text("""
        SELECT *
        FROM models
        WHERE is_active = TRUE
        """)

        # retrieve the dataframe with the model info
        active_model_line = pd.read_sql(query, engine)

        if not active_model_line.empty:
            model_id = active_model_line['model_id'].values[0]
            model_info = retrieve_model_info(database_name, model_id)
            evaluations = retrieve_evaluations(database_name, model_id)
            # Drop the 'created_at' column from the evaluations dataframe so it wont be duplicated
            evaluations = evaluations.drop(columns=['created_at'])
            # Join both dataframes into one
            model_info = pd.merge(model_info, evaluations, on='model_id', how='inner')
            # Select the desired columns
            model_info = model_info[['model_id', \
                                     'model_name', \
                                        'model_type', \
                                            'is_trained', \
                                                'is_active', \
                                                    'model_file_name', \
                                                        'parameters', \
                                                            'created_at', \
                                                                'evaluation_id', \
                                                                    'fp', 'fn', 'tp', 'tn']]
            return model_info
        else:
            return None
    except SQLAlchemyError as e:
            print(f"An error occurred while retrieving the active model info from {database_name}.")
            print(str(e))

def set_model_trained(database_name, model_id):

    try:
            # Create a connection to PostgreSQL database
            engine = create_engine(connection_link(database_name))

            with engine.connect() as connection:

                # Update the value in the table
                query = text(f"""
                            UPDATE models
                            SET 
                            is_trained = TRUE
                            WHERE model_id = :model_id
                            """)
                
                params = {'model_id': model_id}
    
                connection.execute(query, params)
                connection.commit()

                return True # set successfully

    except SQLAlchemyError as e:
            print(f"An error occurred while setting trained in model in {database_name}.")
            print(str(e))
            return False # not set
