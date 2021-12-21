
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import pickle
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join( config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(model_folder, data):
    #read the deployed model and a test dataset, calculate predictions
    y = data.pop('exited')
    y = y.values
    x = data.values
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb") as model:
        lr = pickle.load(model)
    y_pred = list(lr.predict(x))
    return y_pred #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary(data_folder):
    #calculate summary statistics here from datapath
    numeric_columns = ["lastmonth_activity", "lastyear_activity", "number_of_employees", "exited"]
    
    data = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    
    df_summary = []
    for column in numeric_columns:
        df_summary.append([column, "mean", data[column].mean()])
        df_summary.append([column, "median", data[column].median()])
        df_summary.append([column, "standard deviation", data[column].std()])

    return df_summary #return value should be a list containing all summary statistics

##################Function to count missing values
def missing_values(data_folder):
    #calculate percentage of the missing values by columns
    data = pd.read_csv(os.path.join(data_folder, "finaldata.csv"))
    na_pct = list(data.isna().sum(axis=0)/data.shape[0])

    return na_pct

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    
    exec_time = []
    for script in ["training.py" , "ingestion.py"]:
        starttime = timeit.default_timer()
        response = subprocess.run(["python", script])
        timing=timeit.default_timer() - starttime
        exec_time.append([script, timing])
 
    return str(exec_time)#return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(outdated_packages)


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(test_data_path,"testdata.csv")
                          ).set_index('corporation')
    
    model_predictions(model_folder=prod_deployment_path, data=data)
    dataframe_summary(data_folder=dataset_csv_path)
    missing_values(data_folder=dataset_csv_path)
    execution_time()
    outdated_packages_list()





    
