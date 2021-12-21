import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 



##############Function for reporting
def score_model(data_path, model_path, prod_path, plot="confusionmatrix.png"):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    data = pd.read_csv(
        os.path.join(
            data_path,
            "testdata.csv")).set_index('corporation')
    y_data = data.pop('exited')
    y_data = y_data.values
    x_data = data.values
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb") as model:
        lr = pickle.load(model)
    metrics.plot_confusion_matrix(lr, x_data, y_data)
    plt.savefig(os.path.join(model_path, plot))


if __name__ == '__main__':
    score_model(data_path=test_data_path, model_path=output_model_path, prod_path=prod_deployment_path)

