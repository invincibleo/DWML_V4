#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 09.08.17 11:42
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Metrics
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf
import numpy as np
from keras import backend as K

def top3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def sound_event_er(y_true, y_pred):
    K.round()
    K.greater_equal(y_pred, 0.8)
    K.get_value()
    K.set_value()
    return

def apk(actual, predicted):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    actual = keras.backend.get_value(actual)
    predicted = keras.backend.get_value(predicted)
    k = 43

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return keras.backend.variable(score / min(len(actual), k))

def mapk(actual, predicted, k=100):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return keras.backend.mean(tf.metrics.sparse_average_precision_at_k(tf.cast(actual, tf.int64), predicted, 43)[0])
def CCC_V(actual, predicted):
    predicted = predicted[:, 1]
    actual = actual[:, 1]
    pred_mean = K.mean(predicted, axis=0)
    ref_mean = K.mean(actual, axis=0)
    pred_var = K.var(predicted, axis=0)
    ref_var = K.var(actual, axis=0)
    covariance = K.mean((predicted - pred_mean) * (actual - ref_mean), axis=0)
    CCC = (2 * covariance) / (pred_var + ref_var + K.pow((pred_mean - ref_mean), 2))
    return CCC

def CCC_A(actual, predicted):
    predicted = predicted[:, 0]
    actual = actual[:, 0]
    pred_mean = K.mean(predicted, axis=0)
    ref_mean = K.mean(actual, axis=0)
    pred_var = K.var(predicted, axis=0)
    ref_var = K.var(actual, axis=0)
    covariance = K.mean((predicted - pred_mean) * (actual - ref_mean), axis=0)
    CCC = (2 * covariance) / (pred_var + ref_var + K.pow((pred_mean - ref_mean), 2))
    return CCC

def CCC(actual, predicted):
    pred_mean = K.mean(predicted, axis=0)
    ref_mean = K.mean(actual, axis=0)
    pred_var = K.var(predicted, axis=0)
    ref_var = K.var(actual, axis=0)
    covariance = K.mean((predicted - pred_mean) * (actual - ref_mean), axis=0)
    CCC = (2 * covariance) / (pred_var + ref_var + K.pow((pred_mean - ref_mean), 2))
    return K.sum(CCC) / 2
    # truth_all = K.get_value(actual)
    # prediction_all = K.get_value(predicted)
    # dim_num = np.shape(truth_all)[1]
    # score = 0
    # for i in range(dim_num):
    #     truth = truth_all[:, i]
    #     prediction = prediction_all[:, i]
    #     truth = np.reshape(truth, (-1,))
    #     prediction = np.reshape(prediction, (-1,))
    #     pred_mean = np.mean(prediction, -1)
    #     ref_mean = np.mean(truth, -1)
    #
    #     pred_var = np.var(prediction, -1)
    #     ref_var = np.var(truth, -1)
    #
    #     covariance = np.mean(np.multiply((prediction - pred_mean), (truth - ref_mean)), -1)
    #
    #     CCC = (2 * covariance) / (pred_var + ref_var + (pred_mean - ref_mean) ** 2)
    #     score += CCC
    # return keras.backend.variable(score / dim_num)

