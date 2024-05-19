"""
RESTful API to predict the price for any device:
Will take the specs for any device, currently as json file, and send it to my ML model, then return the predicted price.

to run in windows:
open cmd from app.py location:
FLASK_APP=app.py
FLASK_ENV=development
Flask run 
Open up your browser and go to http://127.0.0.1:5000/, and you’ll see the response
"""

from flask import Flask, jsonify
from functions import read_json_request, load_model, predict_sample
from local_paths import path_to_json_file, trained_model_folder
import json

app = Flask(__name__)

device_features = read_json_request(path_to_json_file)
model, scaler = load_model(trained_model_folder)

result, result_class_name = predict_sample(model, scaler, device_features)

with open(path_to_json_file, "r") as file:
                features_dict = json.load(file)
                
device_expected_price = [
features_dict,    
{ "Predicted Price" : result_class_name},
]

@app.get("/")
def get_price():
    return jsonify(device_expected_price)




