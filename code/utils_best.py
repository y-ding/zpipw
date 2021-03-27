import numpy as np
import pandas as pd
import random
import time

from sklearn.linear_model import Ridge, LogisticRegression, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

def get_mse(pred: np.array, truth: np.array) -> float:
    """Get MSE between prediction and groundtruth

    Args:
      pred: predicted array.
      truth: groundtruth array.

    Returns:
      return val: MSE.
    """
    return (np.square(pred - truth)).mean()

def get_score(y_true: np.array, y_pred: list, alpha:float, avg: str) -> list:
    """Get scores between prediction and groundtruth

    Args:    
      y_true: groundtruth binary array.
      y_pred: predicted numerical array.
      alpha: cutoff latency value.

    Returns:
      precision: precision
      recall: recall
      f1: f1
      auc: auc
    """
    if np.array_equal(np.asarray(y_pred), np.asarray(y_pred).astype(bool)):
    	y_b_pred = y_pred
    else:
    	y_b_pred = np.asarray([1 if i >= alpha else 0 for i in y_pred])
    
    # score = precision_recall_fscore_support(y_true, y_b_pred, average=avg, labels=np.unique(y_b_pred))
    # auc = roc_auc_score(y_true, y_b_pred)
    TN, FP, FN, TP = confusion_matrix(y_true, y_b_pred).ravel()
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    # return [score[0], score[1], score[2], auc, TPR, FPR]
    return [TPR*100, FPR*100]


def get_score_all(y_true: np.array, y_pred: list, alpha:float, avg: str) -> list:

    [TPR_gb, FPR_gb] = get_score(y_true, y_pred, alpha, avg)

    return [TPR_gb, FPR_gb]


def get_recall(y_true_real: np.array, y_true: np.array, y_pred: list, alpha:float, bins: list) -> list:
    """Get recall between prediction and groundtruth

    Args:    
      y_true_real: groundtruth real value array.
      y_true: groundtruth binary array.
      y_pred: predicted numerical array.
      bins: bin size interval.

    Returns:
      recall: recall for [90,95), [95, 99), [99+]
    """
    if np.array_equal(np.asarray(y_pred), np.asarray(y_pred).astype(bool)):
    	y_b_pred = y_pred
    else:
    	y_b_pred = np.asarray([1 if i >= alpha else 0 for i in y_pred])

    y_pos_sort = np.argsort(y_true_real.reshape(-1))[(y_true_real.shape[0]-int(sum(y_true))):]
    
    s95 = recall_score(y_true[y_pos_sort[0:bins[0]].tolist()], y_b_pred[y_pos_sort[0:bins[0]].tolist()])
    s99 = recall_score(y_true[y_pos_sort[bins[0]:bins[1]].tolist()], y_b_pred[y_pos_sort[bins[0]:bins[1]].tolist()])
    s99p = recall_score(y_true[y_pos_sort[bins[1]:bins[2]].tolist()], y_b_pred[y_pos_sort[bins[1]:bins[2]].tolist()])
    return [s95*100, s99*100, s99p*100]

def get_recall_all(y_true_real: np.array, y_true: np.array, y_pred: list, alpha:float, bins: list) -> list:

    [s95_gb, s99_gb, s99p_gb] = get_recall(y_true_real, y_true, y_pred, alpha, bins)

    return [s95_gb, s99_gb, s99p_gb]


## Function to get oracle results
def fun_opt(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, y_test_true: np.array,\
    rs: int, delta: float) -> np.array:
    """Get optimal prediction from training and testing data

    Args:    
      X_train: training data.
      Y_train: training numerical groundtruth.
      X_test: testing data.
      Y_test: testing numerical groundtruth.
      y_test_true: testing binary groundtruth.
      rs: random state.
      delta: parameter tuning PS

    Returns:
      p_opt_gb: prediction with gradient boosting
      p_ipw_opt_gb: prediction with gradient boosting
    """
    
    X_all = np.asarray(np.concatenate((X_train, X_test)))
    Y_all = np.asarray(np.concatenate((Y_train, Y_test)))
    y_opt = np.asarray([0] * X_train.shape[0] + y_test_true.tolist())

    start_time = time.time()
    r_opt_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_all, Y_all)
    p_opt_gb = r_opt_gb.predict(X_test)
    tm_opt = time.time() - start_time
 
    ps_start = time.time()
    ## ps
    clf_opt = LogisticRegression(random_state=rs, solver='lbfgs').fit(X_all, y_opt)
    ps_opt = clf_opt.predict_proba(X_all)
    ps1_opt = ps_opt[X_train.shape[0]:,0] + delta
    tm_ps = time.time() - ps_start


    start_time_ipw = time.time()
    p_ipw_opt_gb = p_opt_gb/ps1_opt
    tm_ipw_opt = time.time() - start_time_ipw + tm_ps + tm_opt

    # print("Corr-Oracle time  : {} ms".format(tm_opt/Y_test.shape[0]*1000))
    # print("Causal-Oracle time: {} ms".format(tm_ipw_opt/Y_test.shape[0]*1000))

    return p_opt_gb, p_ipw_opt_gb, tm_opt, tm_ipw_opt

