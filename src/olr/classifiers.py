import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from olr.transformers import ThresholdBinarizer
from sklearn.linear_model import LogisticRegression

__all__ = ['custom_estimator']


class custom_estimator(BaseEstimator, ClassifierMixin):
    """The custom_estimator is en extension of the base
    LogisticRegression

    The custom_estimator has an embedded logistic regression model
    and a ThresholdBinarizer. When the model is being fit, the
    threshold for binary classification is selected
    which minimizes the GINI impurity

    Notes:
    ------
    The Estimator ha the same base signature as LogisticRegression
    with and addition output transformer.

    """

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
        """Fit the model according to the given training data.

                Parameters
                ----------
                X : {array-like, sparse matrix}, shape (n_samples, n_features)
                    Training vector, where n_samples is the number of samples and
                    n_features is the number of features.

                y : array-like, shape (n_samples,)
                    Target vector relative to X.

                sample_weight : array-like, shape (n_samples,) optional
                    Array of weights that are assigned to individual samples.
                    If not provided, then each sample is given unit weight.

                    .. versionadded:: 0.17
                       *sample_weight* support to LogisticRegression.

                Returns
                -------
                self : object

                Notes
                -----
                The SAGA solver supports both float64 and float32 bit arrays.
        """
        self.lrn.fit(X, y, sample_weight)

        self.classes_ = self.lrn.classes_
        self.trb.fit(y, self.lrn.predict_proba(X))

        return self

    def predict(self, X):
        prd = self.trb.transform(self.lrn.predict_proba(X)[:, self.lrn.classes_[1]])
        return prd