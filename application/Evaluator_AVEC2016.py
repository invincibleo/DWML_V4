#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20.11.17 11:50
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Evaluator_AVEC2016
# @Software: PyCharm Community Edition

import numpy as np

class Evaluator_AVEC2016(object):
    def __init__(self):
        self.result_dict = {}

    def evaluate(self, truth_all, prediction_all):
        dim_num = np.shape(truth_all)[1]
        for i in range(dim_num):
            truth = truth_all[:, i]
            prediction = prediction_all[:, i]
            truth = np.reshape(truth, (-1,))
            prediction = np.reshape(prediction, (-1,))
            pred_mean = np.mean(prediction, -1)
            ref_mean = np.mean(truth, -1)

            pred_var = np.var(prediction, -1)
            ref_var = np.var(truth, -1)

            covariance = np.mean(np.multiply((prediction - pred_mean), (truth - ref_mean)), -1)

            CCC = (2 * covariance) / (pred_var + ref_var + (pred_mean - ref_mean) ** 2)
            self.result_dict['CCC of dim' + str(i)] = CCC

    def results(self):
        return self.result_dict

