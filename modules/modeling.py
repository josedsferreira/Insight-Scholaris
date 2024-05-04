import json
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from modules import data
from modules import database
from modules import visualization as vz
from time import time
from imblearn.over_sampling import SMOTE
import featuretools as ft

load_dotenv()
categorical_col_names = os.getenv('CATEGORICAL_COLUMN_NAMES').split(',')

def create_lsvm_model(model_params):
    """
    Create a Linear Support Vector Machine model
    
    Parameters:
    model_params (dict): A dictionary containing the model parameters
    
    Returns:
    model: A Linear Support Vector Machine model
    """
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
    model = XGBClassifier(**model_params)
    return model

def create_randomForest_model(model_params):
    """
    Create a Random Forest model
    
    Parameters:
    model_params (dict): A dictionary containing the model parameters
    
    Returns:
    model: A Random Forest model
    """
    
    model = RandomForestClassifier(**model_params)
    return model

def train_model(database_name, model, model_id, dataset, split, ds_id, fe=False):
    """
    Train a model, update the saved model file and store the evaluation in the database
    
    Parameters:
    database_name (str): The name of the database
    model: A model object
    dataset (pandas dataframe): A dataframe containing the dataset
    model_id (int): The model ID
    split (float): The test size for the train-test split
    ds_id (int): The ID for the dataset used to train the model
    fe (bool): A boolean indicating whether to use feature engineering
    
    Returns:
    bool: True if the model was trained successfully, False otherwise
    """
    try:
        time_start = time()
        #print("0-Dataset Head:")
        #print(dataset.head(3))

        # Split the dataset into features and target
        # X has features minus the target column
        X = dataset.drop(['final_result'], axis=1)
        print("1/11-target droped from dataset")

        if fe:
            # Feature Engineering
            print("X columns: ", X.columns)
            print("X head: ", X.head(3))
            X = data.feature_engineering(X)
            print("1.5-Feature Engineering applied to dataset")

        X = data.create_oneHotEncoder_and_encode(X)
        print("2/11-encoder created and encode applied to dataset")

        # y has the target column and is converted to binary
        y = dataset['final_result']
        print("3/11-target column extracted from dataset into y")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
        print("4/11-dataset split into training and testing sets")

        # Apply SMOTE to the training data
        """ sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print("4.1-SMOTE applied to training data") """

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
        print("5/11-model trained")
        time_fit = time()

        # Predict the response for test dataset
        y_pred = model.predict(X_test)
        print("6/11-model prediction completed")

        # Save the model
        database.update_model_file(database_name, model_id, model)
        print("7/11-model file updated")

        # Store the evaluation in the database
        matrix = confusion_matrix(y_test, y_pred)
        database.store_evaluation(database_name, model_id, matrix)
        print("8/11-evaluation stored in database")

        # set model atribute is_trained to true
        database.set_model_trained(database_name, model_id)
        print("9/11-model trained set to true")

        if database.set_ds_train_id(database_name=database_name, model_id=model_id, ds_id=ds_id):
            print("10/11-dataset train id set")
        
        # Predict the probabilities for the test dataset
        y_score = model.predict_proba(X_test)[:, 1]
        vz.create_ROC(model_id, y_test, y_score)
        vz.create_confusion_matrix(database_name, model_id)
        vz.create_PRC(model_id, y_test, y_score)
        print("11/11-visualizations created")

        time_end = time()
        print(f"Time to fit: {(time_fit - time_start) / 60} minutes")
        print(f"Time elapsed: {(time_end - time_start) / 60} minutes")
        print("-train_model finished")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def get_f1_score(database_name, model_id):
    """
    Get the F1 score of a model
    
    Parameters:
    database_name (str): The name of the database
    model_id (int): The model ID
    
    Returns:
    float: The F1 score of the model
    """
    try:
        # Get the evaluation from the database
        evaluation = database.retrieve_evaluations(database_name, model_id)

        # Calculate the F1 score
        tn = evaluation['tn'].values[0]
        fp = evaluation['fp'].values[0]
        fn = evaluation['fn'].values[0]
        tp = evaluation['tp'].values[0]

        f1_score = tp / (tp + 0.5 * (fp + fn))

        return f1_score
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def create_full_eval(database_name, model_id, pt=False):
    """
    Create a full evaluation of a model
    
    Parameters:
    database_name (str): The name of the database
    
    Returns:
    dict: A dictionary containing the full evaluation of the model
    """

    full_eval = {}

    try:
        # Get the evaluation from the database
        evaluation = database.retrieve_evaluations(database_name, model_id)

        # Calculate the F1 score
        tn = evaluation['tn'].values[0]
        fp = evaluation['fp'].values[0]
        fn = evaluation['fn'].values[0]
        tp = evaluation['tp'].values[0]

        f1_score = tp / float(tp + 0.5 * (fp + fn))
        accuracy = (tp + tn) / float(tp + tn + fp + fn)
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        specificity = tn / float(tn + fp)
        fall_out = fp / float(fp + tn)
        false_discovery_rate = fp / float(fp + tp)
        false_negative_rate = fn / float(fn + tp)
        balanced_accuracy = (recall + specificity) / 2.0

        if pt:
            full_eval['Pontuação F1'] = f1_score
            full_eval['Exatidão'] = accuracy
            full_eval['Precisão'] = precision
            full_eval['Sensibilidade'] = recall
            full_eval['Especificidade'] = specificity
            full_eval['Taxa de Falsos Positivos'] = fall_out
            full_eval['Taxa de Falsa Descoberta'] = false_discovery_rate
            full_eval['Taxa de Falsos Negativos'] = false_negative_rate
            full_eval['Exatidão equilibrada'] = balanced_accuracy
        else:
            full_eval['F1 Score'] = f1_score
            full_eval['Accuracy'] = accuracy
            full_eval['Precision'] = precision
            full_eval['Recall'] = recall
            full_eval['Specificity'] = specificity
            full_eval['Fallout'] = fall_out
            full_eval['False Discovery Rate'] = false_discovery_rate
            full_eval['False Negative Rate'] = false_negative_rate
            full_eval['Ballanced Accuracy'] = balanced_accuracy

        return full_eval
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def predict(database_name, df, df_name):
    """
    Predict the target value of a given dataset
    
    Parameters:
    database_name (str): The name of the database
    df (pandas dataframe): A dataframe containing the dataset
    df_name (str): The name of the dataset
    
    Returns:
    list: A list containing the predicted target values
    """
    try:
        # Load the model
        model_id = int(database.retrieve_active_model_info(database_name)['model_id'][0])
        model = database.retrieve_model(database_name, model_id)
        print("0-model loaded")
        #print("model:", model)

        """ # One Hot Encoding
        df_coded = pd.get_dummies(df, columns=categorical_col_names)
        print("1-get_dummies() applied to dataset")

        # Completes the get_dummies() process on the given DataFrame adding columns for which there was no value in the dataframe.
        df_coded = data.dummies_completer(df_coded)
        print("1.1-dummies_completer() applied to dataset") """

        df_coded = data.oneHotEncode(df)
        print("1-One Hot Encode applied to dataset")

        # Predict the response for the dataset
        y_pred = model.predict(df_coded)
        print("2-model prediction completed")

        # Add predictions as a new column to df
        df['final_result'] = y_pred

        # save the prediction in the database
        df_name = df_name + "_predicted"
        result, df_id = database.store_dataset(db_name=database_name, df=df, df_type=3, df_name=df_name)
        database.update_df_info(database_name, df_id)
        if result:
            print("3-dataset stored in database")
        else:
            print("3- Error: dataset not stored in database")

        return df, df_id
    
    except Exception as e:
        print(f"An error occurred while making a prediction with model {model_id}: {e}")
        return None

def create_gridSearch_rf(database_name, model_name, dataset, ds_id, split):
    """
    Create a Random Forest with grid search

    Parameters:
    database_name (str): The name of the database
    model_name (str): The name of the model
    dataset (pandas dataframe): A dataframe containing the dataset
    ds_id (int): The ID for the dataset used to train the model
    split (float): The test size for the train-test split

    Returns:
    bool: True if the model was trained successfully, False otherwise
    """
    time_start = time()
    try:
        grid = {
            'n_estimators': [int(x) for x in range(100, 1100, 200)],
            'max_depth': [int(x) for x in range(5, 20, 5)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 0.1, 0.2, 0.3, 0.4, 0.5]
        }

        rf = RandomForestClassifier()
        grid_search = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=50, cv=3, n_jobs=-1, verbose=2)

        X = dataset.drop(['final_result'], axis=1)
        print("1-target droped from dataset")

        X = data.create_oneHotEncoder_and_encode(X)
        print("2-encoder created and encode applied to dataset")

        y = dataset['final_result']
        print("3-target column extracted from dataset into y")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
        print("4-dataset split into training and testing sets")
        
        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)
        print("4.9- Fit complete")
        time_fit = time()

        # store model
        model_id = database.store_model(model=grid_search, database_name=database_name, model_name=model_name, model_type=3)
        print("5-model stored in database")

        # Predict the response for test dataset
        y_pred = grid_search.predict(X_test)
        print("6-model prediction completed")

        # Store the evaluation in the database
        matrix = confusion_matrix(y_test, y_pred)
        database.store_evaluation(database_name, model_id, matrix)
        print("8-evaluation stored in database")

        # set model atribute is_trained to true
        database.set_model_trained(database_name, model_id)
        print("9-model trained set to true")

        if database.set_ds_train_id(database_name=database_name, model_id=model_id, ds_id=ds_id):
            print("10-dataset train id set")
        
        # Predict the probabilities for the test dataset
        y_score = grid_search.predict_proba(X_test)[:, 1]
        vz.create_ROC(model_id, y_test, y_score)
        vz.create_confusion_matrix(database_name, model_id)
        vz.create_PRC(model_id, y_test, y_score)
        print("11-visualizations created")

        if database.set_active_model(database_name=database_name, model_id=model_id):
            print('Modelo ativado com sucesso')

        time_end = time()
        print(f"Time to fit: {(time_fit - time_start) / 60} minutes")
        print(f"Time elapsed: {(time_end - time_start) / 60} minutes")
        print("-train_model finished")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def train_model_w_ft(database_name, model, model_id, dataset, split, ds_id):
    """
    Train a model, update the saved model file and store the evaluation in the database
    
    Parameters:
    database_name (str): The name of the database
    model: A model object
    dataset (pandas dataframe): A dataframe containing the dataset
    model_id (int): The model ID
    split (float): The test size for the train-test split
    ds_id (int): The ID for the dataset used to train the model
    
    Returns:
    bool: True if the model was trained successfully, False otherwise
    """
    try:
        time_start = time()

        # Split the dataset into features and target
        # X has features minus the target column
        X = dataset.drop(['final_result'], axis=1)
        print("1-target droped from dataset")

        X = data.create_oneHotEncoder_and_encode(X)
        print("2-encoder created and encode applied to dataset")

        # y has the target column and is converted to binary
        y = dataset['final_result']
        print("3-target column extracted from dataset into y")

        # Feature Engineering
        es = ft.EntitySet(id = 'data')
        print("1/4")

        # Convert column names to strings
        X.columns = X.columns.astype(str)
        print("1.1/4")

        # Add the entire data frame as an entity
        es = es.add_dataframe(dataframe_name='data', dataframe=X, index='index')
        print("2/4")

        # Run deep feature synthesis with transformation primitives
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                            agg_primitives=[],
                                            trans_primitives = ['add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric'])
        print("3/4")
        X = feature_matrix
        print("4/4")
        print("3.1-Feature Engineering applied to dataset")
        print(X.head(3))

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
        print("4-dataset split into training and testing sets")

        # Train the model using the training sets
        model.fit(X_train, y_train)
        print("5-model trained")
        time_fit = time()

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

        if database.set_ds_train_id(database_name=database_name, model_id=model_id, ds_id=ds_id):
            print("10-dataset train id set")
        
        # Predict the probabilities for the test dataset
        y_score = model.predict_proba(X_test)[:, 1]
        vz.create_ROC(model_id, y_test, y_score)
        vz.create_confusion_matrix(database_name, model_id)
        vz.create_PRC(model_id, y_test, y_score)
        print("11-visualizations created")

        time_end = time()
        print(f"Time to fit: {(time_fit - time_start) / 60} minutes")
        print(f"Time elapsed: {(time_end - time_start) / 60} minutes")
        print("-train_model finished")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
