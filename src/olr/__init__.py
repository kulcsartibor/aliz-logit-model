"""
Important classes olr (Optimized Logistic Regression) package:
    - :class:`olr.ThresholdBinarizer`
      The logistic regression threshold optimizer
    - :class:`olr.custom_estimator`
      The main entrypoion of the modified logistic regression package
"""

from __future__ import absolute_import
from olr.transformers import ThresholdBinarizer as ThresholdBinarizer
from olr.classifiers import  custom_estimator as custom_estimator

__all__ = ['ThresholdBinarizer', 'custom_estimator']
