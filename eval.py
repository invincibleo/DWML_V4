# @Time    : 06/02/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : eval.py
# @Software: PyCharm
# Reference: https://stackoverflow.com/questions/45705070/how-to-load-and-use-a-saved-model-on-tensorflow
# https://zhuanlan.zhihu.com/p/31417693

from __future__ import absolute_import, division, print_function

import argparse
import sys
import tensorflow as tf
import data_provider
import numpy as np
from pathlib import Path
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tqdm import tqdm

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt

# tf.enable_eager_execution()

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
    parser.add_argument(
        '--checkpoint_num',
        type=int,
        default=35,
        help="The checkpoint number you want to restore"
    )
    FLAGS, unparsed = parser.parse_known_args()

    output_dir = FLAGS.output_dir
    export_dir = output_dir + '/model_files'
    total_num = 7500 * 9
    seq_length = FLAGS.seq_length
    batch_size = FLAGS.batch_size
    checkpoint_num = FLAGS.checkpoint_num

    ground_truth_all = np.zeros((int(total_num/seq_length/batch_size), batch_size, seq_length, 2))
    prediction_all = np.zeros(ground_truth_all.shape)

    with tf.Session(graph=tf.Graph()) as sess:
        test_ds = data_provider.get_dataset('./tf_records',
                                            is_training=False,
                                            split_name='valid',
                                            batch_size=batch_size,
                                            seq_length=seq_length,
                                            debugging=False)
        iterator = tf.data.Iterator.from_structure(test_ds.output_types,
                                                   test_ds.output_shapes)

        # Get tensor signature from the dataset
        dataset_iter = iterator.get_next()
        test_init_op = iterator.make_initializer(test_ds)

        # Load the model
        model = tf.saved_model.loader.load(sess,
                                           tags=[tag_constants.SERVING],
                                           export_dir=export_dir)

        # Retrieve the tensors
        input_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['audio_input'].name
        audio_input = tf.get_default_graph().get_tensor_by_name(input_name)
        is_training_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['is_training'].name
        is_training = tf.get_default_graph().get_tensor_by_name(is_training_name)
        prediction_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['prediction'].name
        prediction = tf.get_default_graph().get_tensor_by_name(prediction_name)

        # Load the checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, output_dir + "/model.ckpt-" + str(checkpoint_num))

        count_num = 0
        # Testing phase
        try:
            sess.run(test_init_op)
            with tqdm(total=int(total_num / seq_length), desc='Testing') as pbar:
                while True:
                    # Retrieve testing data
                    features_value, ground_truth = sess.run(dataset_iter)
                    features_value = features_value['features']
                    ground_truth = np.array(ground_truth).squeeze(axis=2)

                    prediction_values = sess.run(prediction,
                                                 feed_dict={audio_input: features_value,
                                                            is_training: False})

                    ground_truth_all[count_num, :, :, :] = ground_truth
                    prediction_all[count_num, :, :, :] = prediction_values

                    pbar.update(batch_size)
                    count_num += 1
        except tf.errors.OutOfRangeError:
            ground_truth_all = np.reshape(ground_truth_all, (-1, 2))
            prediction_all = np.reshape(prediction_all, (-1, 2))

plt.plot(ground_truth_all[:, 0])
plt.plot(prediction_all[:, 0])
plt.show()





