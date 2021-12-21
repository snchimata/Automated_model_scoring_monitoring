
import os
import json
import logging
import pandas as pd
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 
    
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
dataset_csv_path = config['output_folder_path']
output_model_path = config['output_model_path']
test_data_path = config['test_data_path']
prod_deployment_path = config['prod_deployment_path']

##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(output_folder_path, "ingestedfiles.txt"), 'r') as file:
    ingested_files = file.read().splitlines()
    
    
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
files = os.listdir(input_folder_path)
input_files = {os.path.join(os.getcwd(),input_folder_path, filename)
               for filename in files if filename.endswith('.csv')}
new_files = set(input_files) - set(ingested_files)


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) > 0:
    logger.info("New files available")
    ingestion.merge_multiple_dataframe(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path)


################## Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data

    with open(os.path.join(prod_deployment_path, "latestscore.txt"), 'r') as f:
        deployed_f1 = float(f.read())
        
    current_f1 = scoring.score_model(
        test_data_path=output_folder_path,
        model_path=prod_deployment_path,
        filename="finaldata.csv")
    logger.info(f"current_f1={current_f1}, deployed_f1={deployed_f1}")
    
    if current_f1 < deployed_f1:
        model_drift = 1
    else:
        model_drift = 0

################## Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
    if model_drift:
        logger.info("Model drift detected")
        training.train_model(
            dataset_csv_path=dataset_csv_path,
            model_path=output_model_path)
        
        scoring.score_model(
            test_data_path=test_data_path,
            model_path=output_model_path)

################## Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
        deployment.store_model_into_pickle(
            model_path=output_model_path,
            prod_path=prod_deployment_path,
            data_path=dataset_csv_path)

        ##################Diagnostics and reporting
        # run diagnostics.py and reporting.py for the re-deployed model
        reporting.score_model(
            plot="confusionmatrix_"+datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".png",
            data_path=test_data_path, 
            model_path=output_model_path, 
            prod_path=prod_deployment_path)
        
        data = pd.read_csv(os.path.join(test_data_path,"testdata.csv")
                          ).set_index('corporation')
        
        model_predictions(model_folder=prod_deployment_path, data=data)
        dataframe_summary(data_folder=dataset_csv_path)
        missing_values(data_folder=dataset_csv_path)
        execution_time()
        outdated_packages_list()







