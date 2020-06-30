import pandas as pd
from classification_model.config import config
import joblib
from sklearn.pipeline import Pipeline




def load_data(*, filename: str) -> pd.DataFrame :

	_data = pd.read_csv(f"{config.DATASET_DIR}/{filename}", encoding = 'ISO-8859-1')
	_data = _data.drop(columns = config.COLUMNS_TO_DROP)
	new_col = _data.columns.str.replace(' ','')
	_data = _data.rename(columns = dict(zip(_data.columns[0:], new_col)))
	_data.dropna(how = 'all', inplace = True)

	return _data


def save_pipeline(*, pipeline_to_persist) -> None:

	save_file_name = f"{config.PIPELINE_SAVE_FILE}.pkl"
	save_path = config.TRAINED_MODEL_DIR / save_file_name
	# remove_old_pipeline(files_to_keep = save_file_name)
	joblib.dump(pipeline_to_persist, save_path)



def load_pipeline(*, filename: str) -> Pipeline:

	file_path = config.TRAINED_MODEL_DIR / filename
	trained_model = joblib.load(filename = file_path)

	return trained_model



# def remove_old_pipeline(*, files_to_keep) -> None:

# 	"""
# 	to make sure we have one to one relationship between
# 	package version and model version
# 	"""

# 	for model_file in config.TRAINED_MODEL_DIR.iterdir():
# 		if model_file.name not in [files_to_keep, '__init__', 'stacked_model.py']:
# 			model_file.unlink()


