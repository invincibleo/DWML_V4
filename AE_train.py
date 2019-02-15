#!/usr/bin/env python
# @Time    : 12/02/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : AE_train.py
# @Software: PyCharm

# Reference: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
# De-convolution: https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0
# Eager_implementation: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb
# Get Global_variable in a graph: https://stackoverflow.com/questions/36533723/tensorflow-get-all-variables-in-scope/36536063
# Check all vars in checkpoints
    # from tensorflow.python.tools import inspect_checkpoint as chkp
    # chkp.print_tensors_in_checkpoint_file(output_dir + "/model.ckpt-" + str(31), tensor_name='', all_tensors=True)

from __future__ import absolute_import, division, print_function

import argparse
import sys
import tensorflow as tf
import data_provider
import losses
import metrics
import models
import numpy as np
from pathlib import Path
from tqdm import tqdm

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

def inference_net(audio_frames=None,
                  hidden_units=256,
                  seq_length=2,
                  num_features=640,
                  latent_dim=50,
                  is_training=False):
    with tf.variable_scope("Encoder"):
        audio_input = tf.reshape(audio_frames, [-1, 640])
        net = tf.layers.Dense(4*num_features,
                              activation=tf.nn.relu)(audio_input)
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.Dense(3*num_features,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.Dense(2*num_features,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.Dense(latent_dim,
                              activation=tf.nn.relu)(net)

        return net

def generative_net(audio_frames=None,
                   hidden_units=256,
                   seq_length=2,
                   num_features=640,
                   latent_dim=50,
                   is_training=False):
    with tf.variable_scope("Decoder"):
        net = tf.reshape(audio_frames, (-1, latent_dim))
        net = tf.layers.Dense(2*num_features,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.Dense(3*num_features,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.Dense(4*num_features,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5)
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.Dense(num_features)(net)
        net = tf.reshape(net, (-1, 1, seq_length, num_features))
        return net


def train(dataset_dir=None,
          init_learning_rate=0.001,
          learning_rate_decay=True,
          batch_size=32,
          seq_length=2,
          num_features=640,
          latent_dim=50,
          epochs=10,
          model_name='e2e_2017',
          output_dir='./output_dir'):

    total_num = 7500 * 9
    loss_list = np.zeros((epochs, int(np.ceil(total_num/seq_length/batch_size))))
    dev_loss_list = np.zeros((epochs, 9)) # 9 files
    g = tf.Graph()
    with g.as_default():
        # Define the datasets
        train_ds = data_provider.get_dataset(dataset_dir,
                                             is_training=True,
                                             split_name='train',
                                             batch_size=batch_size,
                                             seq_length=seq_length,
                                             debugging=False)

        # Pay attension that the validation set should be evaluate file-wise
        dev_ds = data_provider.get_dataset(dataset_dir,
                                           is_training=False,
                                           split_name='valid',
                                           batch_size=int(7500/seq_length),
                                           seq_length=seq_length,
                                           debugging=False)

        # Make the iterator
        iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                                   train_ds.output_shapes)

        # Placeholder for variable is_training
        is_training = tf.placeholder(tf.bool, shape=())

        # Get tensor signature from the dataset
        # features, ground_truth = iterator.get_next()
        # ground_truth = tf.squeeze(ground_truth, 2)
        dataset_iter = iterator.get_next()

        audio_input = tf.placeholder(dtype=tf.float32,
                                     shape=[None, 1, seq_length, num_features],  # Can feed any shape
                                     name='audio_input_placeholder')
        # Get the output tensor
        z = inference_net(audio_frames=audio_input,
                          seq_length=seq_length,
                          num_features=num_features,
                          latent_dim=latent_dim,
                          is_training=is_training)


        x_logit = generative_net(audio_frames=z,
                                 seq_length=seq_length,
                                 num_features=num_features,
                                 latent_dim=latent_dim,
                                 is_training=is_training)

        # PCA
        x_logit_reshaped = tf.reshape(x_logit, (-1, seq_length, num_features))
        x_logit_mean = tf.reduce_mean(x_logit_reshaped, axis=[1], keep_dims=True)
        x_logit_reshaped = x_logit_reshaped - x_logit_mean
        x_logit_covariance = tf.matmul(x_logit_reshaped, x_logit_reshaped, transpose_a=True) / seq_length
        s, u, v = tf.svd(x_logit_covariance)
        pca_low_dim = tf.matmul(tf.reshape(x_logit, (-1, seq_length, num_features)), u[:, :, :latent_dim])
        reconstruction_pca = tf.matmul(pca_low_dim, u[:, :, :latent_dim], transpose_b=True) + x_logit_mean

        tf.summary.audio("reconstruction_audio",
                         tf.reshape(x_logit, (batch_size, -1)),
                         sample_rate=16000,
                         max_outputs=5)
        tf.summary.audio("pca_reconstruction_audio",
                         tf.reshape(reconstruction_pca, (batch_size, -1)),
                         sample_rate=16000,
                         max_outputs=5)

        # Generating power spectrogram
        # stfts = tf.contrib.signal.stft(tf.reshape(audio_input, (batch_size, -1)),
        #                                frame_length=1024,
        #                                frame_step=512,
        #                                fft_length=1024)
        # stfts_reconstruction = tf.contrib.signal.stft(tf.reshape(x_logit, (batch_size, -1)),
        #                                frame_length=1024,
        #                                frame_step=512,
        #                                fft_length=1024)
        #
        # power_spectrograms = tf.real(stfts * tf.conj(stfts))
        # power_spectrograms_re = tf.real(stfts_reconstruction * tf.conj(stfts_reconstruction))
        #
        # power_spectrograms = tf.expand_dims(power_spectrograms, axis=3)
        # power_spectrograms = tf.transpose(power_spectrograms[:, :, :65, :], (0, 2, 1, 3))
        #
        # power_spectrograms_re = tf.expand_dims(power_spectrograms_re, axis=3)
        # power_spectrograms_re = tf.transpose(power_spectrograms_re[:, :, :65, :], (0, 2, 1, 3))
        #
        # tf.summary.image("Power_Spectrogram", power_spectrograms)
        # tf.summary.image("Power_Spectrogram_re", power_spectrograms_re)

        tf.summary.histogram("reconstruction",
                             tf.reshape(x_logit, (-1, num_features)))
        tf.summary.histogram("ground_truth",
                             tf.reshape(audio_input, (-1, num_features)))

        # total_loss = tf.losses.mean_squared_error(predictions=x_logit, labels=audio_input)
        # total_loss = tf.losses.mean_pairwise_squared_error(predictions=x_logit, labels=audio_input)
        total_loss = losses.concordance_cc(prediction=tf.reshape(x_logit, (-1, )),
                                           ground_truth=tf.reshape(audio_input, (-1, )))
        tf.summary.scalar('losses/total_loss', total_loss)

        # # Visualize all weights and bias (takes a lot of space)
        # var_visual_list = []
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     if "kernel" or "bias" in var.name:
        #         var_visual_list.append(var)
        # for var in var_visual_list:
        #     tf.summary.histogram(var.name, var)

        # Learning rate decay
        global_step = tf.Variable(0, trainable=False)
        if learning_rate_decay is True:
            learning_rate = tf.train.cosine_decay_restarts(learning_rate=init_learning_rate,
                                                           global_step=global_step,
                                                           first_decay_steps=50,
                                                           t_mul=2.0,
                                                           m_mul=0.5,
                                                           alpha=0.0,  # Minimum learning rate value as a fraction of the learning_rate.
                                                           name='cosine_decay_restarts')
            add_global = global_step.assign_add(1)
            tf.summary.scalar('learning_rate', learning_rate)
        else:
            learning_rate = init_learning_rate

        # Define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(total_loss)


        # MSE
        with tf.name_scope('my_metrics'):
            mse, mse_update_op = tf.metrics.mean_squared_error(audio_input, x_logit)
        tf.summary.scalar('metric/mse_{}'.format('reconstruction'), mse)

        # Metrics initializer
        metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metrics")
        metrics_vars_initializer = tf.variables_initializer(var_list=metrics_vars)

        with tf.Session(graph=g) as sess:
            # Define the writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(output_dir + '/log/train/', sess.graph)
            val_writer = tf.summary.FileWriter(output_dir + '/log/validation/')
            modal_saver = tf.train.Saver(max_to_keep=20)

            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            sess.run(metrics_vars_initializer)

            # Load the checkpoint for specific variables
            # loading_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Conv2d")
            # modal_saver_loader = tf.train.Saver(var_list=loading_var_list[0:5])
            # modal_saver_loader.restore(sess, output_dir + "/model.ckpt-" + str(199))

            # Save the model, can reload different checkpoint later in testing
            if os.path.exists(output_dir + '/model_files'):
                os.removedirs(output_dir + '/model_files')

            tf.saved_model.simple_save(sess,
                                       export_dir=output_dir + '/model_files',
                                       inputs={"audio_input": audio_input,
                                               "is_training": is_training},
                                       outputs={"z": z,
                                                "x_logit": x_logit})

            # Epochs
            val_old_metric, val_new_metric = [np.inf], [0]
            for epoch_no in range(epochs):
                print('\nEpoch No: {}'.format(epoch_no))
                train_loss, val_loss = 0.0, 0.0
                count_num_train, count_num_dev = 0, 0

                # Initialize the iterator with different dataset
                training_init_op = iterator.make_initializer(train_ds)
                dev_init_op = iterator.make_initializer(dev_ds)

                # Get the learning_rate if learning_rate_decay = True
                if learning_rate_decay is True:
                    _, rate = sess.run([add_global, learning_rate])
                    print('Current learning_rate: {}'.format(rate))

                # Training phase
                try:
                    sess.run(training_init_op)
                    with tqdm(total=int(total_num/seq_length), desc='Training') as pbar:
                        while True:
                            # Retrieve the values
                            features_value, ground_truth = sess.run(dataset_iter)
                            ground_truth = np.array(ground_truth).squeeze(axis=2)
                            features_value = features_value['features']

                            _, loss, summary, _, _, = sess.run((train,
                                                            total_loss,
                                                            merged,
                                                            mse_update_op,
                                                            reconstruction_pca),
                                                           feed_dict={audio_input: features_value,
                                                                      is_training: True})
                            train_loss += loss
                            print("CNM{}".format(loss))
                            loss_list[epoch_no, count_num_train] = loss
                            pbar.update(batch_size)
                            count_num_train += 1
                except tf.errors.OutOfRangeError:
                    train_loss /= count_num_train
                    train_mse = sess.run(mse)
                    sess.run(metrics_vars_initializer)
                    print('Training loss: {}\n'
                          'Training MSE: {}'.format(train_loss,
                                                    train_mse))
                    train_writer.add_summary(summary, epoch_no)

                # Validation phase
                try:
                    sess.run(dev_init_op)
                    with tqdm(total=int(total_num/seq_length), desc='Validation') as pbar_dev:
                        while True:
                            # Retrieve the values
                            features_value, ground_truth = sess.run(dataset_iter)
                            ground_truth = np.array(ground_truth).squeeze(axis=2)
                            features_value = features_value['features']
                            loss, summary, _, _, = sess.run((total_loss,
                                                         merged,
                                                         mse_update_op,
                                                        reconstruction_pca),
                                                        feed_dict={audio_input: features_value,
                                                                   is_training: False})
                            val_loss += loss
                            dev_loss_list[epoch_no, count_num_dev] = loss
                            pbar_dev.update(int(7500/seq_length))
                            count_num_dev += 1
                except tf.errors.OutOfRangeError:
                    val_loss /= count_num_dev
                    val_mse = sess.run(mse)
                    sess.run(metrics_vars_initializer)
                    print('\nEpoch: {}'.format(epoch_no))
                    print('Training loss: {}\n'
                          'Training MSE: {}\n'.format(train_loss,
                                                      train_mse))
                    print('Validation loss: {}\n'
                          'Validation valence MSE: {}\n'.format(val_loss,
                                                                val_mse))
                    val_writer.add_summary(summary, epoch_no)
                    val_new_metric = [val_mse]

                # Have some penalty for the large shoot at beginning
                if val_new_metric <= [x for x in val_old_metric]:
                    # Save the model
                    save_path = modal_saver.save(sess,
                                                 save_path=output_dir + "/model.ckpt",
                                                 global_step=epoch_no,
                                                 )
                    print("Model saved in path: %s" % save_path)
                    val_old_metric = val_new_metric

            # Save the model, can reload different checkpoint later in testing
            if os.path.exists(output_dir + '/model_files'):
                os.removedirs(output_dir + '/model_files')

            tf.saved_model.simple_save(sess,
                                       export_dir=output_dir + '/model_files',
                                       inputs={"audio_input": audio_input,
                                               "is_training": is_training},
                                       outputs={"z": z,
                                                "x_logit": x_logit})

    return loss_list, dev_loss_list


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
        '--epochs',
        type=int,
        default=300,
        help="Epochs number"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help="Initial learning rate"
    )
    parser.add_argument(
        '--learning_rate_decay',
        type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
        default=True,
        help="Initial learning rate"
    )
    FLAGS, unparsed = parser.parse_known_args()

    output_dir = FLAGS.output_dir
    loss_list, dev_loss_list = train(Path(FLAGS.dataset_dir),
                                     init_learning_rate=FLAGS.learning_rate,
                                     learning_rate_decay=FLAGS.learning_rate_decay,
                                     seq_length=FLAGS.seq_length,
                                     batch_size=FLAGS.batch_size,
                                     num_features=640,
                                     latent_dim=FLAGS.latent_dim,
                                     epochs=FLAGS.epochs,
                                     model_name=FLAGS.model,
                                     output_dir=output_dir)
    print(str(loss_list))
    print('\n')
    print(str(dev_loss_list))

    # Save the results
    np.savetxt(output_dir + "/loss_list.txt", loss_list, delimiter=',')
    np.savetxt(output_dir + "/dev_loss_list.txt", dev_loss_list, delimiter=',')

