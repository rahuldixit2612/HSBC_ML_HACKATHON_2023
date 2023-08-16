import requests
import json
import pandas as pd
from data_PREPARED import process_train_data
import pickle as pkl
 
train_data = pd.read_csv("train_data_raw.csv")
train_helper_data = pd.read_csv("train_helper_data_raw.csv")

# Merge train_data and train_helper_data into a single dictionary
data_to_send = {
    "train_data": train_data.to_dict(orient="list"),
    "train_helper_data": train_helper_data.to_dict(orient="list")
}

# Convert the data to JSON format
data_json = json.dumps(data_to_send)

# Send the request to the server
url = "http://192.168.1.7:6060/results"
res = requests.post(url, data=data_json)
out = res.json()
# out = pd.DataFrame(out, columns=list(out.keys()))
# out.to_csv("result.csv", index=False)
print(out)
