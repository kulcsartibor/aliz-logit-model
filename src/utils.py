
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np


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