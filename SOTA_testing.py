#!/usr/bin/env python
# @Time    : 14/03/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : testing.py
# @Software:

from __future__ import absolute_import, division, print_function

import argparse
import sys
import json
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
import shutil

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

def get_ds(dataset, batch_size=32):
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=10000)
    return dataset

if __name__ == "__main__":

    flags = tf.flags
    # Learning related FLAGS
    flags.DEFINE_boolean("is_training", True, "True for training, False for testing [Train]")
    flags.DEFINE_float("learning_rate", .0001, "Learning rate [.0001]")
    flags.DEFINE_float("prob_use_near_non_neighber", 0.1, "Prob to use neaer non neigher as pair [0.5]")
    flags.DEFINE_integer("batch_size", 1, "The size of batch.(Training) [1000]")
    flags.DEFINE_integer("val_batch_size", -1, "The size of batch. (Validation) If -1 then same as batch_size [-1]")
    flags.DEFINE_boolean("is_learning_rate_decay", True, "If use learning rate decay [True]")
    flags.DEFINE_string("decay_type", "cosine_decay_restarts", "Which decay type to use [cosine_decay_restarts]")
    flags.DEFINE_integer("epochs", 100, "The training epochs [100]")
    flags.DEFINE_integer("ds_repeat_num", 0, "The times the ds will repeat for one epoch [0]")
    flags.DEFINE_boolean("batch_norm", False,
                         "Use of batch norm [False] (always False for discriminator if gradient_penalty > 0)")

    # Data set and path related FLAGS
    flags.DEFINE_string("dataset", "CAMIL", "The name of the dataset chose to run")
    flags.DEFINE_string("name", "CAMIL", "The name of dataset for creating folder name puposes [CAMIL, VAST]")
    flags.DEFINE_string("feature_type", "ild", "The name of which type of feature [ild, ipd, bin]")
    flags.DEFINE_integer("p_num", 50, "p_num indicate the percentage of the training set is used [50]")
    flags.DEFINE_string("data_dir", "./data", "Directory containing datasets [./data]")
    flags.DEFINE_string("log_dira", "log", "Directory name to save the tensorboard logs [logs_mmd]")
    flags.DEFINE_string("logdir_root", None, "Log dir root [None]")
    flags.DEFINE_string("restore_from", None,
                        "Directory in which to restore the model from. Cannot use with --logdir[None]")
    flags.DEFINE_string("model_files_dir", "model_files", "Directory name to save the model files [model_files]")
    flags.DEFINE_string("output_dir", "./", "Directory name to save all the results [./]")
    flags.DEFINE_string("suffix", '', "For additional settings ['', '_tf_records']")
    flags.DEFINE_integer("checkpoint_num", -1, "Checkpoint num that want to load [50]")
    flags.DEFINE_integer("checkpoint_every", 500, "Save checkpoints every [500] iterations")
    flags.DEFINE_integer("max_checkpoints", 5, "How many checkpoints to save maximumly. [5]")
    flags.DEFINE_boolean("visualize_weights_bias", False, "If to visualize weights and bias in tensorboard")
    flags.DEFINE_boolean("visual_test_embeddings", False, "If to visualize embeddings tensorboard")
    flags.DEFINE_boolean("save_checkpoint", False, "If to save the checkpoint and model")

    # Model related
    flags.DEFINE_string("architecture", "dense1n", "The name of the architecture [dense1n, dense1w, dense2w]")
    flags.DEFINE_string("loss_type", "pvoc", "Type of loss, pvoc or contrastive [pvoc, contrastive]")
    flags.DEFINE_integer("num_features", -1, "Dimension of the feature input. -1 to infer the num[400]")
    flags.DEFINE_integer("embedding_size", 3, "The size of the embeddings [3]")
    flags.DEFINE_string("model", "siamese", "The model type [siamese, cramer, wgan_gp]")
    flags.DEFINE_float("similarity_margin_1", 4.0,
                       "How much degree difference in pairs will be considered as pair [4.0]")
    flags.DEFINE_float("similarity_margin_2", -1,
                       "How much degree difference in pairs will be considered as pair, in seconde dimension, -1 is equal to dimension 1 [-1]")
    flags.DEFINE_integer("dim_for_similar_pairs", -1,
                         "use which dimension of the azimuth(0) or elevation(1) or both (-1) [-1]")
    flags.DEFINE_boolean("save_use_max_val_metric", True, "Save a checkpoint when the validation metric is bigger")
    flags.DEFINE_integer("num_rand_embeddings", 2000, "The number of the embeddings to visualize [1000]")
    flags.DEFINE_integer("early_stop_patience", 50,
                         "The number of epochs that the validation performance doesn't increase before early stopping [10]")
    flags.DEFINE_boolean("is_weighted_mu", True, "Is the mu in loss weighted with original distance [True]")
    flags.DEFINE_float("dropout_in", 0.3, "How much dropout after the input [0.3]")
    flags.DEFINE_float("dropout_m", 0.3, "How much droput in the middle layers [0.3]")

    # WaveNet Model
    flags.DEFINE_integer("seq_length", 320000, "Sequence length. [0]")
    flags.DEFINE_boolean("histograms", False, "Whether to save histograms in the summary [False]")
    flags.DEFINE_integer("gc_channels", 0,
                         "Number of channels in (embedding size) of global conditioning vector. None indicates there is no global conditioning.[0]")
    flags.DEFINE_integer("lc_channels", 0,
                         "Number of channels in (embedding size) of local conditioning vector. None indicates there is no local conditioning.[0]")
    flags.DEFINE_float("l2_regularization_strength", 0, "Coefficient in the L2 regularization. [0]")
    flags.DEFINE_float("silence_threshold", 0.3, "Threshold to remove the silence. [0.3]")
    flags.DEFINE_string("optimizer", 'adam', "Optimizer. [adam, sgd, rmsprop]")
    flags.DEFINE_float("momentum", 0.9, "Used by sgd or rmsprop optimizer. Ignored by the Adam. [0.3]")
    flags.DEFINE_float("input_noise_std", 0.1, "Data augmentation, input noise std. [0.1]")
    flags.DEFINE_integer("num_steps", 50000, "Numbers of iterations. [100000]")
    flags.DEFINE_string("wavenet_params", "./wavenet_params_stacked.json", "WaveNet parameter JSON path.")
    flags.DEFINE_boolean("store_metadata", False,
                         "Whether to store advanced debugging information.(execution time, memory consumption) for use with tensorboard [False]")

    # Machine settings
    flags.DEFINE_boolean("log", True, "Wheather to write log to a file in samples directory [True]")
    flags.DEFINE_boolean("seprate", False, "Calculate CCC seperately on each file [False]")
    flags.DEFINE_integer("threads", 18, "Upper limit for number of threads [np.inf]")
    flags.DEFINE_float("gpu_mem", .4, "GPU memory fraction limit [0.9]")

    # Running flags
    flags.DEFINE_integer("samples_per_label", 0, "How many samples per label, will be updated in the program. [0]")
    flags.DEFINE_integer("output_dims", 0,
                         "How many output dimensions, normally in order [Arousal, Valence, Liking], will be updated in the program. [0]")
    flags.DEFINE_string("session_name", "", "What is the session class name, will be updated in the program. ")
    flags.DEFINE_string("chk_file_dir", "", "where the checkpoint are saved")
    args = flags.FLAGS


    def cal_ccc(prediction, gs, seprate=False):
        if seprate is False:
            prediction = np.reshape(prediction, (-1))
            gs = np.reshape(gs, (-1))
            num_labels_per_file = np.shape(gs)[-1]

            pred_var = np.var(prediction, axis=-1)
            pred_mean = np.mean(prediction, axis=-1)

            gs_var = np.var(gs, axis=-1)
            gs_mean = np.mean(gs, axis=-1)

            cov = np.sum((prediction - pred_mean) * (gs - gs_mean)) / (num_labels_per_file - 1)

            denominator = (pred_var + gs_var + np.square(pred_mean - gs_mean))
            print("Calculating CCC as a whole!")
            return np.mean((2 * cov) / (denominator + 1e-6))
        else:
            prediction = np.reshape(prediction, (9, 7500))
            gs = np.reshape(gs, (9, 7500))
            num_labels_per_file = np.shape(gs)[-1]

            pred_var = np.var(prediction, axis=-1)
            pred_mean = np.mean(prediction, axis=-1)

            gs_var = np.var(gs, axis=-1)
            gs_mean = np.mean(gs, axis=-1)

            cov = np.array([np.sum((prediction[x, :] - pred_mean[x])*(gs[x, :] - gs_mean[x]))/(num_labels_per_file - 1) for x in range(int(np.shape(gs)[0]))])

            denominator = (pred_var + gs_var + np.square(pred_mean - gs_mean))
            print("Calculating CCC seperately!")
            return np.mean((2 * cov) / (denominator + 1e-6))

    def int_or_None(input):
        return (None if input == 0 else input)

    lr = 'lr%.5f' % args.learning_rate
    dataset_desp = 'dataset%s_%s%d' % (args.dataset,
                                       args.feature_type,
                                       args.p_num)
    description = args.chk_file_dir

    model_dir = os.path.abspath(os.path.join(args.output_dir, description))
    assert os.path.exists(model_dir), print("model dir not exist: " + model_dir)

    if args.dataset == 'AVEC2016_RECOLA':
        import data_provider
        num_features = 640
        args.output_dims = 2
        ds_valid = data_provider.get_dataset(args.data_dir,
                                             is_training=False,
                                             split_name='valid',
                                             batch_size=args.batch_size,
                                             seq_length=args.seq_length,
                                             debugging=False)
        iterator = tf.data.Iterator.from_structure(ds_valid.output_types, ds_valid.output_shapes)
        iter_get_next = iterator.get_next()
        ds_init_op = iterator.make_initializer(ds_valid)
        # ds_test = data_provider.get_dataset(args.data_dir,
        #                                     is_training=False,
        #                                     split_name='test',
        #                                     batch_size=args.batch_size,
        #                                     seq_length=args.seq_length,
        #                                     debugging=False)

        import models
        # batch_size
        # histograms
        # gc_channels
        # global_condition_cardinality
        # l2_regularization_strength
        sess = tf.Session()

        audio_batch = tf.placeholder(tf.float32, shape=(None, 1, None, num_features), name='input')
        labels_holder = tf.placeholder(tf.float32,
                                       shape=(None, None, args.output_dims),
                                       name='labels')
        is_training = tf.placeholder(tf.bool, shape=())

        predictions_logit, extra_loss = models.e2e_2018_provide(audio_frames=audio_batch,
                                                                hidden_units=256,
                                                                seq_length=args.seq_length,
                                                                batch_size=args.batch_size,
                                                                num_features=num_features,
                                                                number_of_outputs=args.output_dims,
                                                                is_training=is_training)

        variables_to_restore = {
            var.name[:-2]: var for var in tf.global_variables()
            if not ('state_buffer' in var.name or 'pointer' in var.name)}
        saver = tf.train.Saver(variables_to_restore)

        # Load the model
        print('Restoring model from {}'.format(model_dir))
        # saver.restore(sess, model_dir)

        # Load the checkpoint
        if args.checkpoint_num == -1:
            print('Load the model!')
            ckpt = tf.train.get_checkpoint_state(model_dir)
            saver.restore(sess, save_path=ckpt.all_model_checkpoint_paths[-1])
        else:
            saver.restore(sess, save_path=model_dir + "/model.ckpt-" + str(args.checkpoint_num))

        npy_output_dir = os.path.join(model_dir, 'npy')
        # Validation set
        count_num = 0
        total_num_points = 7500 * 9 // args.batch_size
        ground_truth_all = np.zeros((total_num_points, args.batch_size, args.output_dims))
        prediction_all = np.zeros((total_num_points, args.batch_size, args.output_dims))
        try:
            sess.run(ds_init_op)
            while True:
                # TODO: multiple datasets support
                features_value, labels = sess.run(iter_get_next)
                features_value = np.reshape(features_value, (-1, args.seq_length, 1, num_features))
                labels = np.reshape(labels[:, :, :, :args.output_dims], (-1,
                                                                         args.seq_length,
                                                                         args.output_dims))

                prediction_values = sess.run([predictions_logit],
                                             feed_dict={audio_batch: features_value,
                                                        labels_holder: labels,
                                                        is_training: False})
                # Unwrap from list
                prediction_values = prediction_values[0]
                ground_truth_all[count_num, :, :] = labels
                prediction_all[count_num, :, :] = prediction_values
                count_num += 1
        except tf.errors.OutOfRangeError:
            ground_truth_all = np.reshape(ground_truth_all, (-1, args.output_dims))
            prediction_all = np.reshape(prediction_all, (-1, args.output_dims))

            ground_truth_all = ground_truth_all[0:7500*9, :]
            prediction_all = prediction_all[0:7500*9, :]

            ground_truth_all = np.reshape(ground_truth_all, (9, 7500, args.output_dims))
            prediction_all = np.reshape(prediction_all, (9, 7500, args.output_dims))

            ar_ccc = cal_ccc(prediction_all[:, :, 0], ground_truth_all[:, :, 0], seprate=args.seprate)
            va_ccc = cal_ccc(prediction_all[:, :, 1], ground_truth_all[:, :, 1], seprate=args.seprate)

            print("Arousal_CCC:{}, Valence_CCC:{}".format(ar_ccc, va_ccc))
            np.save(npy_output_dir + '/Pred_val_sq_{}k_a{}_v{}.npy'.format(args.seq_length//1000,
                                                                           int(np.ceil(ar_ccc * 10000)),
                                                                           int(np.ceil(va_ccc * 10000))),
                    [ground_truth_all, prediction_all, ar_ccc, va_ccc])


        # testing set
        # count_num = 0
        # total_num_points = int(np.ceil(7500 * 9 / (args.seq_length //args.samples_per_label)))
        # ground_truth_all = np.zeros((total_num_points, (args.seq_length //args.samples_per_label), args.output_dims))
        # prediction_all = np.zeros((total_num_points, (args.seq_length //args.samples_per_label), args.output_dims))
        # # testing training set phase
        # try:
        #     ds_test.init_dataset(sess=sess)
        #     while True:
        #         # TODO: multiple datasets support
        #         features_value, _ = sess.run(ds_test.iter_get_next)
        #         features_value = np.reshape(features_value, (-1, ds_test.seq_length, 1))
        #         labels = np.zeros((np.shape(features_value)[0], ds_test.seq_length // args.samples_per_label, args.output_dims))
        #
        #         prediction_values = sess.run([predictions_logit],
        #                                      feed_dict={audio_batch: features_value,
        #                                                 labels_holder: labels,
        #                                                 is_training: False})
        #         # Unwrap from list
        #         prediction_values = prediction_values[0]
        #         gt_size = labels.shape[1]
        #         pre_size = prediction_values.shape[0]
        #         ground_truth_all[count_num, :gt_size, :] = labels
        #         prediction_all[count_num, :pre_size, :] = prediction_values
        #         count_num += 1
        # except tf.errors.OutOfRangeError:
        #     ground_truth_all = np.reshape(ground_truth_all, (-1, args.output_dims))
        #     prediction_all = np.reshape(prediction_all, (-1, args.output_dims))
        #
        #     ground_truth_all = ground_truth_all[0:7500*9, :]
        #     prediction_all = prediction_all[0:7500*9, :]
        #
        #     ground_truth_all = np.reshape(ground_truth_all, (9, 7500, args.output_dims))
        #     prediction_all = np.reshape(prediction_all, (9, 7500, args.output_dims))
        #
        #     ar_ccc = cal_ccc(prediction_all[:, :, 0], ground_truth_all[:, :, 0])
        #     va_ccc = cal_ccc(prediction_all[:, :, 1], ground_truth_all[:, :, 1])
        #
        #     print("Arousal_CCC:{}, Valence_CCC:{}".format(ar_ccc, va_ccc))
        #     np.save(npy_output_dir + '/Pred_test_sq_{}k_a{}_v{}.npy'.format(args.seq_length//1000,
        #                                                                     int(np.ceil(ar_ccc * 10000)),
        #                                                                     int(np.ceil(va_ccc * 10000))),
        #             [ground_truth_all, prediction_all, ar_ccc, va_ccc])


