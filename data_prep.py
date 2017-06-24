""" Processing data for mercedes kaggle contest """
#!/usr/bin/env
import pandas as pd
import numpy as np
import helper_funcs as helpers
from mercedes_classes import DataFrameSelector, Dummifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


class Pipes():
    def identity_pipe(self,columns):
        identity_pipeline = Pipeline([
                ('selector', DataFrameSelector(columns)),
            ])
        return identity_pipeline

    def base_pipe(self,columns):
        base_pipeline = Pipeline([
                ('selector', DataFrameSelector(columns)),
                ('std_scaler', StandardScaler()),
            ])
        return base_pipeline

    def dummy_pipe(self, columns):
        dummy_pipeline = Pipeline([
                ('dummies', Dummifier()),
                ('selector', DataFrameSelector(columns)),
                ('std_scaler', StandardScaler()),
            ])
        return dummy_pipeline

    def pca_sep_pipe(self, columns, reduce_to):
        pca_sep_pipeline = Pipeline([
                ('dummies', Dummifier()),
                ('selector', DataFrameSelector(columns)),
                ('std_scaler', StandardScaler()),
                ('pca', PCA(n_components=reduce_to))
            ])
        return pca_sep_pipeline

class DataPrep():
    def __init__(self, base_pipe=None, dummy_pipe=None, pca_sep_pipe=None, drop_cols=None, separate_cols=None):
        self.drop_cols = drop_cols
        self.base_pipe = base_pipe
        self.dummy_pipe = dummy_pipe
        self.pca_sep_pipe = pca_sep_pipe
        self.separate_cols = separate_cols

    def load_data(self, trainpath, testpath):
        if self.drop_cols:
            train = helpers.load_data(trainpath, drop_cols=drop_cols+['ID'])
            test = helpers.load_data(testpath, drop_cols=drop_cols)
        else:
            train = helpers.load_data(trainpath, drop_cols=['ID'])
            test = helpers.load_data(testpath)
        train_target = train["y"].values
        train = train.drop(["y"], axis=1)
        test_id = test["ID"].values
        test = test.drop(["ID"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(train, train_target, test_size=0.2, random_state=42)
        return X_train, y_train, X_test, y_test, test, test_id

    def transform(self, X_train, X_test, test):
        pipe = Pipes()
        if self.base_pipe:
            columns = helpers.common_cols(X_train, test)
            base_pipeline = pipe.base_pipe(columns)
            X_train,X_test, test = helpers.apply_pipe(X_train, X_test, test, base_pipeline)
            return X_train, X_test, test

        elif self.dummy_pipe:
            columns = helpers.cat_cols(X_train, test)
            dummy_pipeline = pipe.dummy_pipe(columns)
            X_train,X_test, test = helpers.apply_pipe(X_train, X_test, test, dummy_pipeline)
            return X_train, X_test, test

        elif self.pca_sep_pipe:
            X_train_sep = X_train[self.separate_cols]
            X_test_sep = X_test[self.separate_cols]
            test_sep = test[self.separate_cols]
            X_train = X_train.drop(self.separate_cols, axis=1)
            X_test= X_test.drop(self.separate_cols, axis=1)
            test = test.drop(self.separate_cols, axis=1)

            columns = helpers.cat_cols(X_train_sep, test_sep)
            pca_sep_pipeline = pipe.pca_sep_pipe(columns, 10)
            X_train_sep,X_test_sep, test_sep = helpers.apply_pipe(X_train_sep, X_test_sep, test_sep, pca_sep_pipeline)

            columns_nocat = helpers.common_cols(X_train, test)
            base_pipeline = pipe.base_pipe(columns_nocat)
            X_train,X_test, test = helpers.apply_pipe(X_train, X_test, test, base_pipeline)
            
            X_train = np.concatenate((X_train, X_train_sep), axis=1)
            X_test= np.concatenate((X_test, X_test_sep), axis=1)
            test = np.concatenate((test, test_sep), axis=1)
            return X_train, X_test, test
        else:
            columns = helpers.common_cols(X_train, test)
            identity_pipe = pipe.identity_pipe(columns)
            X_train,X_test, test = helpers.apply_pipe(X_train, X_test, test, identity_pipe)
            return X_train, X_test, test
            

