""" Classes to preprocess data for mercedes kaggle contest """
#!/usr/bin/env

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ Transforms a pandas dataframe to a numpy array 
    to feed to other transformations. """
    def __init__(self, attribute_names=None):
        """ Initializing with column names of interest """
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        """ Takes a pandas dataframe, checks if columns have
        been specified and transform to numpy array """
        if self.attribute_names:
            df_cols = X.columns.tolist()
            missing_cols = [x for x in self.attribute_names if x not in df_cols]
            total_cols = list(set(self.attribute_names + missing_cols))
            if missing_cols:
                for col in missing_cols:
                    X[col] = np.zeros(X.shape[0])
            return X[total_cols].values
        else:
            return X.values

class Dummifier(BaseEstimator, TransformerMixin):
    """ One hot encoding for categorical columns """
    def __init__(self, cat=True):
        """ Initializing with a flag for 
        categorical columns """
        self.cat = cat
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        """ Returns dataframe with dummy columns"""
        dataframe = pd.get_dummies(X)
        return dataframe
