import random
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from olr.classifiers import custom_estimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


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
#    else:
#        print('Confusion matrix, without normalization')

#    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


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


def ROC(X, y, c, r):
    # makes cross_validation for given parameters c,r. Returns FPR, TPR (averaged)
    dic_weight = {1: len(y) / (r * np.sum(y)), 0: len(y) / (len(y) - r * np.sum(y))}
    lrn = LogisticRegression(penalty='l2', C=c, class_weight=dic_weight)

    N = 5  # how much k-fold
    N_iter = 3  # repeat how often (taking the mean)
    mean_tpr = 0.0
    mean_thresh = 0.0
    mean_fpr = np.linspace(0, 1, 50000)

    for it in range(N_iter):
        skf = StratifiedKFold(n_splits=N, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            lrn.fit(X_train, y_train)
            y_prob = lrn.predict_proba(X_test)[:, lrn.classes_[1]]

            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_thresh += np.interp(mean_fpr, fpr, thresholds)
            mean_tpr[0] = 0.0

    mean_tpr /= (N * N_iter)
    mean_thresh /= (N * N_iter)
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr, roc_auc_score(y_test, y_prob), mean_thresh


def plot_roc(X,y, list_par_1, par_1 = 'C', par_2 = 1):

    f = plt.figure(figsize = (12,8));
    for p in list_par_1:
        if par_1 == 'C':
            c = p
            r = par_2
        else:
            r = p
            c = par_2
        list_FP, list_TP, AUC, mean_thresh = ROC(X, y, c, r)
        plt.plot(list_FP, list_TP, label = 'C = {}, r = {}, TPR(3e-4) = {:.4f}, AUC = {:.4f}'.format(c,r,list_TP[10],AUC));
    plt.legend(title = 'values', loc='lower right')
    plt.xlim(0, 0.001)   #we are only interested in small values of FPR
    plt.ylim(0.5, 0.9)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC detail')
    plt.axvline(3e-4, color='b', linestyle='dashed', linewidth=2)
    plt.show()
    plt.close()


if __name__ == "__main__":
    random.seed(a=None, version=2)
    np.random.seed(seed=0)

    df = pd.read_csv("../input/creditcard.csv")
    print(df.head(3))

    y = np.array(df.Class.tolist())

    df = df.drop('Class', 1)
    df = df.drop('Time', 1)
    df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

    X = np.array(df.values)

    print("Fraction of frauds: {:.5f}".format(np.sum(y) / len(y)))

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

    # plot_roc(X, y, [1, 3, 10, 70], 'r', 1)
    #
    # plot_roc(X, y, [0.001, 0.01, 0.1, 1, 100], 'C', 1)

    sys.exit(0)

