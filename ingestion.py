import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path,output_folder_path):
    #check for datasets, compile them together, and write to an output file
    
    
    #Load and process data
    csv_files_list = glob.glob(os.path.join(os.getcwd(),input_folder_path,"*.csv"))
    df = pd.concat(map(pd.read_csv,csv_files_list),ignore_index=True)
    df.drop_duplicates(inplace=True)
    
    #Save data
    output_path = os.path.join(os.getcwd(), output_folder_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "a") as report_file:
        for line in csv_files_list:
            report_file.write(line + "\n")
            
    df.to_csv(os.path.join(output_path,"finaldata.csv"),
              index=False)



if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path,output_folder_path)
