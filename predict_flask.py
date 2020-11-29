from flask import Flask
from flask import request
from model_package import ml_model
import pickle
import pandas as pd
import json

# Create flask app to receive json data and make rain predictions
app = Flask('predicteapp')

@app.route('/')
@app.route('/predict', methods=['POST'])
def predict():
    """Given list of json records, clean and predict whether it will rain"""
    json_data_str = request.get_json()
    json_data = json.loads(json_data_str)
    predictions = ml_model.predict_rain(json_data)
    # Return {'predictions':...,'probability':...} json
    return predictions

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
