import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def calculate_split_gini(y, p, thr):
    """Calculates the GINI impurity of a split with given probabilities
    and a threshold

    :param y: class value (known)
    :param p: probability of class predicted
    :param thr: the threshold
    :return: split GINI index
    """
    y_left_i = p[:, 0] < thr
    y_right_i = np.logical_not(y_left_i)

    gini_l = calculate_gini(y[y_left_i]) * np.sum(y_left_i) / (p.size)
    gini_r = calculate_gini(y[y_right_i]) * np.sum(y_right_i) / (p.size)

    return gini_l + gini_r


def calculate_gini(y):
    """Calculates the GINI impurity value of
    list of samples.

    :param y: array of classes
    :return: GINI value
    """
    uniqueValues, occurCount = np.unique(y, return_counts=True)

    n_sample = y.size

    gini = 0

    for i in np.nditer(occurCount):
        gini += (i/n_sample) * (1 - i/n_sample)

    return gini


class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    """Transforms a logistic regression output probability to a modified model.

        This transformer takes y and p output pairs of a logistic regression model
        and adjusts the threshold to minimize the GINI impurity of the split.

        The transformation is calculated as::

            X_scaled = scale * X + min - X.min(axis=0) * scale
            where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

        Parameters
        ----------
        :param resolution : float, default(0.01) - 1%
            this value determines the steps which is used in the search. The threshold
            is calculated up to this level

        Notes
        -----
        This transformer works only with binary classifiers
        """

    def __init__(self, resolution=0.01):
        self.resulotion = resolution
        self.threshold = -1

    def fit(self, y, y_p):
        """Fits the transformer models to determine the
        logistic regression threshold which minimizes the GINI impurity

        :param y: know classes of samples
        :param y_p: predicted probabily of class=0
        :return: self
        """
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
        """Transforms the estimated  logistic regression output into
        to classes.

        Note:
        -----
        This transform model expects the probability of class 0 as input

        :param y: probabily of class 0
        :return: predicted class. [0,1]
        """
        cond = y > self.threshold
        not_cond = np.logical_not(cond)
        y[cond] = 1
        y[not_cond] = 0

        return y