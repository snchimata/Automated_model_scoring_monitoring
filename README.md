# Automated model scoring and monitoring
# Udacity Dynamic risk assessment system
author: Sai Chimata
date: December 12, 2021

The goal of this project is to build an automated model training, scoring and monitoring system.

## Files
1. ingestion.py - To ingest multiple datasets into a dataframe.
2. training.py - To train the final data with a logistic regression model.
3. scoring.py - To generate an F1 Score for the trained logistic regression model against testdata.csv under testdata folder.
4. deployment.py - To record the production model pkl file and production model metrics including model scores, confusion matrix plot.
5. diagnostics.py - To generate feature statistic , prediction summary, check missing values, and to calculate the execution time of running processes
6. reporting.py - To report model performance by generating confusion matrix
7. app.py and apicalls.py - To create an api and to verify real-time responses.
8. fullprocess.py - For future model maintenance and monitoring, new data processes and model drift

## Instruction
1. Run ingestion.py
2. Run training.py
3. Run scoring.py
4. Run deployment.py
5. Run fullprocess.py

## Note
Project is set up for periodic model re-training using cronjob every 10 minutes. If a model drift is detected on new data, retraining, redeploying, diagnostics, and reporting will run accordingly.
