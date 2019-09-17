import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from olr.transformers import ThresholdBinarizer
from sklearn.linear_model import LogisticRegression

__all__ = ['custom_estimator']


class custom_estimator(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None, output_transformer=None):
        self.classes_ = None

        self.lrn = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                      class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class,
                                      verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

        if output_transformer is None:
            self.trb = ThresholdBinarizer()
        else:
            self.trb = output_transformer

    def fit(self, X, y, sample_weight=None):
        self.lrn.fit(X, y, sample_weight)

        self.classes_ = self.lrn.classes_
        self.trb.fit(y, self.lrn.predict_proba(X))

        return self

    def predict(self, X):
        prd = self.trb.transform(self.lrn.predict_proba(X)[:, self.lrn.classes_[1]])
        return prd