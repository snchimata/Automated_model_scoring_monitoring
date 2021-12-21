import os
import json
import requests

#Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000"

with open('config.json','r') as f:
    config = json.load(f)
    
output_model_path = config['output_model_path']
test_data_path = os.path.join(config['test_data_path'])

#Call each API endpoint and store the responses
def get_responses(URL):
    response1 = requests.post(URL+'/prediction',json={
            "dataset_path": os.path.join(
                test_data_path,
                "testdata.csv")}).text
    response2 = requests.get(URL+'/scoring').text
    response3 = requests.get(URL+'/summarystats').text
    response4 = requests.get(URL+'/diagnostics').text

    #combine all API responses
    responses = [response1, response2, response3, response4]
    return responses
    

if __name__ == "__main__":
    output_model_path = config['output_model_path']
    test_data_path = os.path.join(config['test_data_path'])
    responses = get_responses(URL)
    
    with open(os.path.join(output_model_path, "apireturns.txt"), "w") as returns_file:
        returns_file.write(str(responses))



