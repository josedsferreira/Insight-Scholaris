import json
import pandas as pd
from dotenv import load_dotenv
import os
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