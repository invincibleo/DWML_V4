# @Time    : 06/02/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : eval.py
# @Software: PyCharm
from __future__ import absolute_import, division, print_function

import argparse
import sys
import tensorflow as tf
import data_provider
from pathlib import Path
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./e2e_2018',
        help='Path to output dir where everything logs in'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=25,
        help="Batch size"
    )
    parser.add_argument(
        '--seq_length',
        type=int,
        default=150,
        help="sequence length"
    )
    FLAGS, unparsed = parser.parse_known_args()

    output_dir = FLAGS.output_dir
    export_dir = output_dir + '/model_files'
    with tf.Session(graph=tf.Graph()) as sess:
        dev_ds = data_provider.get_dataset('./tf_records',
                                           is_training=False,
                                           split_name='valid',
                                           batch_size=FLAGS.batch_size,
                                           seq_length=FLAGS.seq_length,
                                           debugging=False)
        iterator = tf.data.Iterator.from_structure(dev_ds.output_types,
                                                   dev_ds.output_shapes)

        # Get tensor signature from the dataset
        features, ground_truth = iterator.get_next()
        ground_truth = tf.squeeze(ground_truth, 2)

        dev_init_op = iterator.make_initializer(dev_ds)
        sess.run(dev_init_op)

        model = tf.saved_model.loader.load(sess,
                                           tags=[tag_constants.SERVING],
                                           export_dir=export_dir)
        # input_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['features'].name
        # features = tf.get_default_graph().get_tensor_by_name(input_name)

        is_training_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['is_training'].name
        is_training = tf.get_default_graph().get_tensor_by_name(is_training_name)

        prediction_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['prediction'].name
        prediction = tf.get_default_graph().get_tensor_by_name(prediction_name)

