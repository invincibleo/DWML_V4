#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 17:06
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Learner
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import tensorflow as tf
import shutil
import os
import sys
from keras.models import model_from_json

from abc import ABCMeta
from abc import abstractmethod

class Learner(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        self.learner_name = kwargs.get('learner_name', '')
        self.dataset = kwargs.get('dataset', None)
        self.FLAGS = kwargs.get('flag', None)
        self.hash_name_hashed = ''
        self.input_shape = tuple([int(x) for x in self.FLAGS.dimension.split(',')])
        self.setup_keras()

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def generate_pathes(self):
        self.hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(self.FLAGS.__str__())).hexdigest()
        print(self.FLAGS.__str__())
        model_json_file_addr = "tmp/model/" + str(self.hash_name_hashed) + "/model.json"
        model_h5_file_addr = "tmp/model/" + str(self.hash_name_hashed) + "/model.h5"
        return model_json_file_addr, model_h5_file_addr

    def copy_configuration_code(self):
        if not os.path.exists("tmp/model/" + str(self.hash_name_hashed)):
            code_base_addr = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

            os.makedirs("tmp/model/" + str(self.hash_name_hashed))
            os.makedirs('tmp/model/' + str(self.hash_name_hashed) + '/checkpoints/')
            shutil.copytree(code_base_addr + '/application/', 'tmp/model/' + str(self.hash_name_hashed) + '/application/')

    def load_model(self, model):
        model.load_weights("tmp/model/" + self.hash_name_hashed + "/model.h5")
        return model

    def load_model_from_file(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()

        # load json and create model
        json_file = open(model_json_file_addr, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(model_h5_file_addr)
        print("Loaded model from disk")
        return model

    def save_model(self, hist, model):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()

        # Saving the objects:
        with open('tmp/model/objs.txt', 'wb') as histFile:  # Python 3: open(..., 'wb')
            # pickle.dump([hist, model], f)
            for key, value in hist.history.iteritems():
                histFile.write(key + '-' + ','.join([str(x) for x in value]))
                histFile.write('\n')

        # serialize model to JSON
        model_json = model.to_json()
        with open(model_json_file_addr, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_h5_file_addr)
        print("Saved model to disk")

    def setup_keras(self):
        # Threading
        thread_count = 20
        os.environ['GOTO_NUM_THREADS'] = str(thread_count)
        os.environ['OMP_NUM_THREADS'] = str(thread_count)
        os.environ['MKL_NUM_THREADS'] = str(thread_count)

        if thread_count > 1:
            os.environ['OMP_DYNAMIC'] = 'False'
            os.environ['MKL_DYNAMIC'] = 'False'
        else:
            os.environ['OMP_DYNAMIC'] = 'True'
            os.environ['MKL_DYNAMIC'] = 'True'

        # Select Keras backend
        os.environ["KERAS_BACKEND"] = 'tensorflow'
        os.environ["HIP_VISIBLE_DEVICES"] = '0'