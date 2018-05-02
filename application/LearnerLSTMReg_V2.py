#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16.11.17 10:12
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : LearnerLSTMReg
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import tensorflow as tf
import shutil
import math
import random
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz
from core.Models import *
from core.Learner import Learner
from core.Metrics import *


def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


class LearnerLSTMReg(Learner):
    def learn(self):
        model_json_file_addr, model_h5_file_addr = self.generate_pathes()
        continue_training = False
        if not os.path.exists(model_json_file_addr) or continue_training:
            self.copy_configuration_code()  # copy the configuration code so that known in which condition the model is trained

            dataset = train_input_fn(features=self.dataset.training_total_features,
                                     labels=self.dataset.training_total_labels,
                                     batch_size=self.FLAGS.train_batch_size)
            dataset_iter = dataset.make_one_shot_iterator()

            sess = tf.InteractiveSession()

            is_training = tf.placeholder(tf.bool, (), name='training')
            learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')
            time_steps = self.input_shape[0]
            num_features = self.input_shape[1]
            batch_size = self.FLAGS.train_batch_size
            lstm_size = 4096

            x = tf.placeholder(tf.float32, [batch_size, time_steps, num_features], name='x')
            labels = tf.placeholder(tf.float32, [batch_size, 2], name='labels')

            # Design the network (execute one of the following cells)
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            initializer = tf.keras.initializers.glorot_normal(seed=1000)
            hidden_state = tf.Variable(initializer([batch_size, lstm_size]))
            current_state = tf.Variable(initializer([batch_size, lstm_size]))
            state = hidden_state, current_state

            np_x, np_labels = dataset_iter.get_next()

            input = tf.unstack(np_x, time_steps, 1)
            outputs, state = tf.nn.static_rnn(lstm_cell, input, initial_state=state, dtype="float32")

            # x = tf.layers.dense(outputs, 4096)
            # x = tf.nn.relu(x)
            # x = tf.layers.batch_normalization(x, training=is_training)
            # x = tf.layers.dense(x, 2048)
            # x = tf.nn.relu(x)
            # logits = tf.layers.batch_normalization(x, training=is_training)

            logits = tf.layers.dense(outputs[-1], units=2, activation=None)

            loss = tf.reduce_mean(tf.squared_difference(tf.cast(np_labels, tf.float32), logits))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            # Set up tensorboard
            writer = tf.summary.FileWriter('./tmp/logs/tensorboard/')
            writer.add_graph(sess.graph)

            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(var.name.replace(':', '_'), var)

            tf.summary.scalar('validation_loss', tf.reduce_mean(loss))

            merged_summary = tf.summary.merge_all()

            # Allocate memory and initialize variables in the network:
            sess.run(tf.global_variables_initializer())
            for step in range(self.FLAGS.epochs):
                sess.run(optimizer,
                         feed_dict={
                             # x: np_x,
                             # labels: np_labels,
                             is_training: True,
                             learning_rate: self.FLAGS.learning_rate
                         })

                if step % 100 == 0:
                    np_dev_logits = sess.run(logits, {x: dataset.validation.images, is_training: False})
                    print(np.mean(np.argmax(np_dev_logits, 1) == dataset.validation.labels))

                    summary_to_write = sess.run(
                        merged_summary,
                        {x: dataset.validation.images,
                         labels: dataset.validation.labels,
                         is_training: False
                         })
                    writer.add_summary(summary_to_write, step)

            # np_test_logits = sess.run(logits, {x: dataset.test.images})
            # np.mean(np.argmax(np_test_logits, 1) == dataset.test.labels)

    def predict(self):
        model = self.load_model_from_file()

        model_json_file_addr, model_h5_file_addr = self.generate_pathes()

        # load weights into new model
        model.load_weights(model_h5_file_addr)

        predictions_all = model.predict(self.dataset.validation_total_features,
                                        batch_size=self.FLAGS.test_batch_size,
                                        verbose=0)

        Y_all = self.dataset.validation_total_labels

        Y_all = np.reshape(Y_all, (-1, np.shape(Y_all)[-1]))
        predictions_all = np.reshape(predictions_all, (-1, np.shape(predictions_all)[-1]))

        return Y_all, predictions_all
