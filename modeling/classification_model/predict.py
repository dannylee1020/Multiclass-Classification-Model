import pandas as pd
import numpy as np

from classification_model.config import config
from classification_model.train_pipeline import encode_target
from classification_model.processing.data_management import load_pipeline, load_data

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import logging


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}.pkl"
_status_pipe = load_pipeline(filename = pipeline_file_name)



def make_prediction(*, input_data):
	data = pd.read_json(input_data)
	predictions = _status_pipe.predict(data)
	results = {'predictions' : predictions} 

	return results



if __name__ == '__main__':
	data = load_data(filename = config.DATA_FILE)
	X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES], data[config.TARGET],
														test_size = 0.4, random_state = 42)
	y_pred = _status_pipe.predict(X_test)
	y_test = encode_target(y_test)
	acc = accuracy_score(y_test, y_pred)

	print(f"test_accuracy: {acc}")
	print(classification_report(y_test, y_pred))



