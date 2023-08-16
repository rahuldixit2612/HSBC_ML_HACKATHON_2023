from flask import Flask, request, jsonify
import pandas as pd
import pickle as pkl
from data_PREPARED import process_train_data
from inference import Inference

app = Flask(__name__)

def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pkl.load(model_file)
    return model

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    data_train = data["train_data"]
    data_train_helper = data["train_helper_data"]
    data_train = pd.DataFrame.from_dict(data_train)
    data_train_helper = pd.DataFrame.from_dict(data_train_helper)

    data_preparation = process_train_data(data_train, data_train_helper)
      
    inference = Inference(data_preparation)
    result = inference.prediction()
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6060)
