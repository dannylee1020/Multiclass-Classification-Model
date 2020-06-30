from flask import Blueprint, request, jsonify
from classification_model.predict import make_prediction
from classification_model.processing.data_management import load_data
from classification_model.config import config as model_config
import json


prediction_app = Blueprint('prediction_app', __name__)

_data = load_data(filename = 'test.csv')
features = _data[model_config.FEATURES]




@prediction_app.route('/test', methods = ['GET'])
def test():
	if request.method == 'GET':
		return 'test running: looking good!'




@prediction_app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		json_data = request.get_json()
		results = make_prediction(input_data = json_data)
		predictions = results.get('predictions').tolist()

		return jsonify({'predictions' : predictions})




# @prediction_app.route('/predict', methods = ['GET'])
# def predict():
# 	if request.method == 'GET':
# 		_data = features.to_json(orient = 'records')
# 		results = make_prediction(input_data = _data)
# 		predictions = results.get('predictions').tolist()

# 		return jsonify({'predictions' : predictions[:10]})





