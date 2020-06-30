
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import classification_model.processing.preprocessing as pp
from classification_model.config import config

from classification_model.trained_model import stacked_model as sm



status_pipeline = Pipeline(

	[	
		('fill na', 
		pp.FillNA(variables = config.FEATURES)),

		# ('clean funding amount',
		# pp.CleanFundingAmount(variables = config.FUNDING_VAR, group_var = config.GROUP_VAR)),	

		# ('categorical encoder',
		# pp.CategoricalEncoder(variables = config.CATEGORICAL_VAR)),

		('scaler', MinMaxScaler()),
		# ('PCA', PCA(n_components = config.PCA_DIMENSION)),
		('stacked_model', sm.stacked_model(base_models = config.BASE_MODELS, meta_model = config.META_MODEL, n_folds = config.NFOLD))
	]
)