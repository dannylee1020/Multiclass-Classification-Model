import pathlib
# import classification_model

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

import classification_model


PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_model"
DATASET_DIR = PACKAGE_ROOT / "datasets"


# pipeline
PIPELINE_NAME = 'stacked_model'
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output"


# models
BASE_MODELS = [('model_1', RandomForestClassifier(n_estimators = 150)), ('model_2', LGBMClassifier(n_estimators = 150))]
META_MODEL = LogisticRegression()
NFOLD = 5



# data
DATA_FILE = 'investments_VC.csv'
TARGET = 'status'


# variables to remove
COLUMNS_TO_DROP = ['permalink','name','homepage_url','category_list','state_code',
                   'region','founded_at','founded_month','founded_quarter']

# funding variables
FUNDING_VAR = 'funding_total_usd'
GROUP_VAR = 'market'


FEATURES = ['founded_year', 'seed', 'venture', 'equity_crowdfunding',
	       'undisclosed', 'grant', 'private_equity', 'post_ipo_debt', 'round_A',
	       'round_B', 'round_C', 'round_D']

# # categorical variable
# CATEGORICAL_VAR = ['market', 'country_code', 'city', 'first_funding_at',
#        			  'last_funding_at']


# # numerical variable
# NUMERICAL_VAR = ['funding_total_usd', 'funding_rounds', 'founded_year', 'seed',
# 		       	 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note',
# 		         'debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity',
# 		         'post_ipo_debt', 'secondary_market', 'product_crowdfunding', 'round_A',
# 		         'round_B', 'round_C', 'round_D', 'round_E', 'round_F', 'round_G',
# 		         'round_H']


# # features
# FEATURES = ['funding_total_usd', 'funding_rounds', 'founded_year', 'seed',
# 	       	 'venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note',
# 	         'debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity',
# 	         'post_ipo_debt', 'secondary_market', 'product_crowdfunding', 'round_A',
# 	         'round_B', 'round_C', 'round_D', 'round_E', 'round_F', 'round_G',
# 	         'round_H','market','country_code', 'city', 'first_funding_at',
# 			 'last_funding_at']



# PCA dimensions
PCA_DIMENSION = 300




