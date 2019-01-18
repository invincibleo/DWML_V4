# @Time    : 17/01/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : train_v2.py
# @Software: PyCharm

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import data_provider

from tensorflow import keras
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

def train_input_fn():
    dataset_dir = "./test_output_dir/tf_records"
    dataset = data_provider.get_dataset(dataset_dir,
                                        is_training=True,
                                        split_name='train',
                                        batch_size=32,
                                        seq_length=2,
                                        debugging=False)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def my_model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,     # An instance of tf.estimator.ModeKeys
    params):  # Additional configuration

    labels = labels[:, 0, 0, :]
    labels = tf.reshape(labels, [-1, 2])
    # Use `input_layer` to apply the feature columns.
    # net = tf.feature_column.input_layer(features, params['feature_columns'])
    net = tf.reshape(features['features'], [-1, 2*640])

    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    predicted_classes = logits
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

    mse = {'mse': tf.metrics.mean_squared_error(labels=labels,
                                        predictions=predicted_classes,
                                        name='acc_op')}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=mse)

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=mse)


def train(dir):
    my_feature_columns = tf.feature_column.numeric_column(key='features')

    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 2,
        },
        model_dir='./logs')

    classifier.train(input_fn=train_input_fn, steps=10000)

if __name__ == "__main__":
  train(Path("./test_output_dir"))
