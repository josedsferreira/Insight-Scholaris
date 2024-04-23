import json
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from modules import data
from modules import database

def create_lsvm_model(model_params):
    """
    Create a Linear Support Vector Machine model
    
    Parameters:
    model_params (dict): A dictionary containing the model parameters
    
    Returns:
    model: A Linear Support Vector Machine model
    """
    from sklearn.svm import LinearSVC
    model = LinearSVC(**model_params)
    return model

def create_xgboost_model(model_params):
    """
    Create an XGBoost model
    
    Parameters:
    model_params (dict): A dictionary containing the model parameters
    
    Returns:
    model: An XGBoost model
    """
    from xgboost import XGBClassifier
    model = XGBClassifier(**model_params)
    return model

def train_model(database_name, model, model_id, dataset, split):
    """
    Train a model, update the saved model file and store the evaluation in the database
    
    Parameters:
    database_name (str): The name of the database
    model: A model object
    dataset (pandas dataframe): A dataframe containing the dataset
    model_id (int): The model ID
    split (float): The test size for the train-test split
    
    Returns:
    bool: True if the model was trained successfully, False otherwise
    """
    try:
        #print("0-Dataset Head:")
        #print(dataset.head(3))

        # Split the dataset into features and target
        # X has features minus the target column
        X = dataset.drop(['final_result'], axis=1)
        print("1-target droped from dataset")

        # Check if get_dummies() has already been applied
        # should not be needed
        if any(X.dtypes == 'object') or any(X.dtypes == 'category'):
            X = pd.get_dummies(X)
            print("2-get_dummies() applied to dataset")

        # y has the target column and is converted to binary
        y = dataset['final_result'] #.map({'Pass': 1, 'Fail': 0}) # mapping ja nao deve ser necessario
        print("3-target column extracted from dataset into y")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
        print("4-dataset split into training and testing sets")

        #print("Head of X_train:")
        #print(X_train.head(3))
        #print("X Dtypes")
        #print(X_train.dtypes)
        #print("Head of y_train:")
        #print(y_train.head(3))
        #print("Y Dtypes")
        #print(y_train.dtypes)

        # Train the model using the training sets
        model.fit(X_train, y_train)
        print("5-model trained")

        # Predict the response for test dataset
        y_pred = model.predict(X_test)
        print("6-model prediction completed")

        # Save the model
        database.update_model_file(database_name, model_id, model)
        print("7-model file updated")

        # Store the evaluation in the database
        matrix = confusion_matrix(y_test, y_pred)
        database.store_evaluation(database_name, model_id, matrix)
        print("8-evaluation stored in database")

        # set model atribute is_trained to true
        database.set_model_trained(database_name, model_id)
        print("9-model trained set to true")

        print("10-train_model finished")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
