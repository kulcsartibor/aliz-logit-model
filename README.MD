# Modified Logistic regression model

This package contains a modified logistic regression model which has an additional
optimization. The threshold is optimized to minimize the GINI impurity

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install initialize
the python environment with the required packages.

```bash
pip install -r requirements.txt
```

## Usage

```python
from olr.classifiers import custom_estimator

lrn = custom_estimator()

lrn.fit(X_train, y_train)
y_pred = lrn.predict(X_test)
```

## Run test script

From the src directory run the following command:

```bash
python logit_heart_desease.py
```