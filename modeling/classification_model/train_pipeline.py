from classification_model.pipeline import status_pipeline
from classification_model.processing.data_management import load_data, save_pipeline
from classification_model.config import config

from sklearn.model_selection import train_test_split
import numpy as np
import logging

_logger = logging.getLogger(__name__)

def encode_target(data):
	X = data.copy()
	X = X.replace('NaN', np.nan)
	X = X.fillna('missing')
	X = X.map({'missing': 0, 'operating': 1, 'acquired': 2, 'closed': 3})
	return X


def run_training() -> None:

	data = load_data(filename = config.DATA_FILE)
	X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES], data[config.TARGET], test_size = 0.4, random_state = 42)
	y_train = encode_target(y_train)
	status_pipeline.fit(X_train, y_train)
	save_pipeline(pipeline_to_persist = status_pipeline)



if __name__ == "__main__":
	run_training()