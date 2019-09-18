import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def calculate_split_gini(y, p, thr):
    y_left_i = p[:, 0] < thr
    y_right_i = np.logical_not(y_left_i)

    gini_l = calculate_gini(y[y_left_i]) * np.sum(y_left_i) / (p.size)
    gini_r = calculate_gini(y[y_right_i]) * np.sum(y_right_i) / (p.size)

    return gini_l + gini_r


def calculate_gini(y):
    uniqueValues, occurCount = np.unique(y, return_counts=True)

    n_sample = y.size

    gini = 0

    for i in np.nditer(occurCount):
        gini += (i/n_sample) * (1 - i/n_sample)

    return gini


class ThresholdBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, resolution=0.01):
        self.resulotion = resolution
        self.threshold = -1
        # print("Initializing classifier:\n
        # args, _, _, values = inspect.getargvalues(inspect.currentframe())
        # values.pop("self")
        #
        # for arg, val in values.items():
        #     setattr(self, arg, val)

    def fit(self, y, y_p):
        thr_range = np.arange(self.resulotion, 1, self.resulotion)

        gini = float('inf')

        for t in np.nditer(thr_range):
            new_gini = calculate_split_gini(y, y_p, t)
            if new_gini < gini:
                gini = new_gini
                self.threshold = t

        print("The deducted threshols is: " + str(self.threshold))

        return self

    def transform(self, y):
        cond = y > self.threshold
        not_cond = np.logical_not(cond)
        y[cond] = 1
        y[not_cond] = 0

        return y

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return True if x >= self.threshold else False




