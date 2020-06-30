
from classification_model.predict import make_prediction
from classification_model.processing.data_management import load_data
from classification_model.config import config
import json


def test_make_single_prediction():
	# Given
	test_data = load_data(filename = 'test.csv')
	single_test = test_data[config.FEATURES][0:1]
	single_test_json = single_test.to_json(orient = 'records')


	# When 
	subject = make_prediction(input_data = single_test_json)

	# Then
	assert subject is not None
	assert subject[0] == 1
 


def test_make_multiple_predictions():
	# Given 
	test_data = load_data(filename = 'test.csv')
	original_data_length = len(test_data)
	multiple_test = test_data[config.FEATURES]
	multiple_test_json = multiple_test.to_json(orient = 'records')

	# when
	subject = make_prediction(input_data = multiple_test_json)

	# Then
	assert subject is not None
	assert len(subject) == 33149


