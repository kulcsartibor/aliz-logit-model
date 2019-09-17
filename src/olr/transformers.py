import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class ThresholdBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, lrn:LogisticRegression):
        self.lrn = lrn
        # print("Initializing classifier:\n
        # args, _, _, values = inspect.getargvalues(inspect.currentframe())
        # values.pop("self")
        #
        # for arg, val in values.items():
        #     setattr(self, arg, val)

    def fit(self, X, y):

        gs = GridSearchCV()

        self.trh = 0.6

        return self

    def transform(self, X):
        cond = X > self.trh
        not_cond = np.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0

        return X

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return True if x >= self.trh else False


