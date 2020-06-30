import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingClassifier


from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone

class stacked_model(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, base_models = None, meta_model = None, n_folds = None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self,X,y):
        level0 = []
        for name, model in self.base_models:
            level0.append((name, model))
        level1 = self.meta_model
        self.get_stacking_ = StackingClassifier(estimators = self.base_models, final_estimator = level1, cv = self.n_folds)
        self.get_stacking_.fit(X,y)
        
        return self
    
    def predict(self, X):
        y_pred = self.get_stacking_.predict(X)
        return y_pred

