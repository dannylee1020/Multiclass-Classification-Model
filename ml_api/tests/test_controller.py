from classification_model.processing.data_management import load_data
from classification_model.config import config as model_config
import json



def testing_test_endpoint_returning_200(flask_test_client):
	# When
	response = flask_test_client.get('/test')

	# Then
	assert response.status_code == 200



def testing_prediction_endpoint_returns_prediction(flask_test_client):
	# Given
	test_data = load_data(filename = 'test.csv')
	features = test_data[model_config.FEATURES][0:10]
	features_json = features.to_json(orient = 'records')

	# When
	response = flask_test_client.post('/predict', json = features_json)

	# Then
	response_json = json.loads(response.data)
	prediction = response_json['predictions']
	pred_len = len(prediction)
	assert response.status_code == 200
	assert pred_len == 10
