from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import *
from scoring import *


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    data = pd.read_csv(request.json.get('dataset_path')
                       ).set_index('corporation')
    
    predictions = model_predictions(
        model_folder=prod_deployment_path, data=data)
    return str(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1 = score_model(
        test_data_path=test_data_path,
        model_path=prod_deployment_path,
        filename="testdata.csv")
    return str(f1)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary_stats = dataframe_summary(data_folder=dataset_csv_path)
    return  str(summary_stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    na_pct = missing_values(data_folder=dataset_csv_path)
    exec_time = execution_time()
    op = outdated_packages_list()
    #return str((na_pct, exec_time, op))
    return  json.dumps({"execution_time" : exec_time , "missing_data": na_pct , "outdated_packages": op})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
