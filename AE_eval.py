#!/usr/bin/env python
# @Time    : 15/02/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : AE_eval.py
# @Software: PyCharm

# Reference: https://stackoverflow.com/questions/45705070/how-to-load-and-use-a-saved-model-on-tensorflow
# https://zhuanlan.zhihu.com/p/31417693
# Embedding visualization: https://www.cnblogs.com/cloud-ken/p/9329703.html
# https://zhuanlan.zhihu.com/p/38001015
# https://www.jianshu.com/p/d5339d04aa17

from __future__ import absolute_import, division, print_function

import argparse
import sys
import tensorflow as tf
import data_provider
import numpy as np
from pathlib import Path
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector
# tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./tf_records',
        help='Path to the tensorflow records dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./2017_e2e_output_dir',
        help='Path to output dir where everything logs in'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='e2e_2017',
        help="Model name that you want to use to train"
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
        '--latent_dim',
        type=int,
        default=256,
        help="Dimension of the latent mean or logvar"
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
    latent_dim = FLAGS.latent_dim
    checkpoint_num = FLAGS.checkpoint_num

    ground_truth_all = np.zeros((int(np.ceil(total_num/seq_length/batch_size)), batch_size, seq_length, 2))
    prediction_all = np.zeros(ground_truth_all.shape)
    z_all = np.zeros((int(np.ceil(total_num/seq_length/batch_size)), batch_size, seq_length, latent_dim))

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
        x_logit_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['x_logit'].name
        x_logit = tf.get_default_graph().get_tensor_by_name(x_logit_name)
        z_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['z'].name
        z = tf.get_default_graph().get_tensor_by_name(z_name)

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

                    prediction_values = sess.run(z,
                                                 feed_dict={audio_input: features_value,
                                                            is_training: False})

                    ground_truth_all[count_num, :, :, :] = ground_truth
                    # prediction_all[count_num, :, :, :] = prediction_values
                    z_all[count_num, :, :, :] = np.reshape(prediction_values, (-1, seq_length, latent_dim))

                    pbar.update(batch_size)
                    count_num += 1
        except tf.errors.OutOfRangeError:
            ground_truth_all = np.reshape(ground_truth_all, (-1, 2))
            prediction_all = np.reshape(prediction_all, (-1, 2))
            z_all = np.reshape(z_all, (-1, latent_dim))

            ground_truth_all = ground_truth_all[0:total_num, :]
            prediction_all = prediction_all[0:total_num, :]
            z_all = z_all[0:total_num, :]

            rand_idx = np.random.randint(z_all.shape[0], size=5000)
            # Visualization of the embeddings
            summary_writer = tf.summary.FileWriter(output_dir + '/embeddings')
            embedding_var = tf.Variable(z_all[rand_idx, :], name='latent_z')
            sess.run(embedding_var.initializer)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = "metadata.tsv"
            projector.visualize_embeddings(summary_writer, config)
            # Write labels
            embeddings_metadata_addr = output_dir + '/embeddings/metadata.tsv'
            if not os.path.exists(output_dir + '/embeddings'):
                os.makedirs(output_dir + '/embeddings')
            with open(embeddings_metadata_addr, 'w') as f:
                f.write("Index\tArousal\tValence\n")
                idx = 0
                for gt in ground_truth_all[rand_idx, :]:
                    idx += 1
                    f.write("{}\t{}\t{}\n".format(idx, int(gt[0]*10), int(gt[1]*10)))

            saver = tf.train.Saver()  # Save the model files into the same folder of embeddings
            saver.save(sess, os.path.join(output_dir + '/embeddings', "embeding_model.ckpt"), 1)

