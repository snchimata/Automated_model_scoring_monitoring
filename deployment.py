from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def store_model_into_pickle(model_path=model_path,
        prod_path=prod_deployment_path,
        data_path=dataset_csv_path):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    # Prepare file paths
    file_paths = [
        (os.path.join(
            data_path, "ingestedfiles.txt"), os.path.join(
            prod_path, "ingestedfiles.txt")), (os.path.join(
                model_path, "trainedmodel.pkl"), os.path.join(
                    prod_path, "trainedmodel.pkl")), (os.path.join(
                        model_path, "latestscore.txt"), os.path.join(
                            prod_path, "latestscore.txt"))]

    # copy files to production
    if not os.path.exists(prod_path):
        os.mkdir(prod_path)
    for file_path in file_paths:
        prac, prod = file_path
        os.system(f'cp {prac} {prod}')
        
        
        
if __name__ == '__main__':
    store_model_into_pickle(
        model_path=model_path,
        prod_path=prod_deployment_path,
        data_path=dataset_csv_path
    )
