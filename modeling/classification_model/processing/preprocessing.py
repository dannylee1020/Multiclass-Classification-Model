import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from classification_model.config import config



class FillNA(BaseEstimator, TransformerMixin):

	def __init__(self, variables = None):
		self.variables = variables

	def fit(self, X,y = None):

		return self

	def transform(self, X):
		X = X.copy()
		for feature in self.variables:
			if X[feature].dtypes == 'O':
				X[feature] = X[feature].replace('NaN', np.nan)
				X[feature] = X[feature].fillna('missing')
			else:
				X[feature] = X[feature].fillna(0)

		return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):

	def __init__(self, variables = None):
		self.variables = variables

	def fit(self, X,y = None):

		return self

	def transform(self, X):
		X = X.copy()
		X = pd.get_dummies(X[self.variables], drop_first = True, columns = config.CATEGORICAL_VAR)

		return X


class CleanFundingAmount(BaseEstimator, TransformerMixin):

	def __init__(self, variables = None, group_var = None):
		if not isinstance(variables, list):
			self.variables = [variables]
		else:
			self.variables = variables

		self.group_var = group_var

	def fit(self,X,y = None):

		return self

	def transform(self, X):
		X = X.copy()
		for feature in self.variables:
			X[feature] = X[feature].str.replace(' ','')

			temp1 = X[X[feature] == '-']
			temp2 = X[X[feature] != '-']

			temp1[feature] = temp1[feature].apply(lambda x: x.replace('-','0'))
			temp2[feature] = temp2[feature].apply(lambda x: x.replace(',',''))

			combined = pd.concat([temp1, temp2])
			X_new = combined.sort_index()

			X_new[feature] = X_new[feature].astype(float)
			X_new[feature] = X_new[feature].replace(0,np.nan)
			X_new[feature] = X_new[feature].fillna(X_new.groupby(config.GROUP_VAR)[feature].transform('mean'))

		return X_new











