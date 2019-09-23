import random
import sys

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from olr.classifiers import custom_estimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def show_data(cm, print_res = 0):
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    if print_res == 1:
        print('Precision =     {:.3f}'.format(tp/(tp+fp)))
        print('Recall (TPR) =  {:.3f}'.format(tp/(tp+fn)))
        print('Fallout (FPR) = {:.3e}'.format(fp/(fp+tn)))
    return tp/(tp+fp), tp/(tp+fn), fp/(fp+tn)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    random.seed(a=None, version=2)
    np.random.seed(seed=0)

    df = pd.read_csv("../input/heart.csv")
    print(df.head(3))

    y = np.array(df.target.tolist())

    df = df.drop('target', 1)

    X = np.array(df.values)

    print("Fraction of target: {:.5f}".format(np.sum(y) / len(y)))

    # lrn = LogisticRegression()
    lrn = custom_estimator()
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        break

    lrn.fit(X_train, y_train)
    y_pred = lrn.predict(X_test)

    print

    cm = confusion_matrix(y_test, y_pred)
    if lrn.classes_[0] == 1:
        cm = np.array([[cm[1, 1], cm[1, 0]], [cm[0, 1], cm[0, 0]]])

    plot_confusion_matrix(cm, ['0', '1'], )
    pr, tpr, fpr = show_data(cm, print_res=1)

    sys.exit(0)

