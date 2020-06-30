
import pandas as pd
import numpy as np
import pathlib
import classification_model

PATH = pathlib.Path(classification_model.__file__).resolve().parent
DATASET = PATH / "datasets"
FILE_NAME = 'investments_VC.csv'
_data = pd.read_csv(f"{DATASET}/{FILE_NAME}", encoding = 'ISO-8859-1')

test_size = np.ceil(len(_data)*0.3).astype(int)
test_data = _data.iloc[test_size:,:]
test_data = test_data.drop(columns = 'status')
test_data.to_csv(f"{DATASET}/test.csv", index = False)
