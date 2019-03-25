#!/usr/bin/env python
# @Time    : 13/03/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : siamese_train.py
# @Software: PyCharm

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
import shutil
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# import matplotlib
# matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
# import matplotlib.pyplot as plt


tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# aaa = (np.reshape(np.linspace(0, 7500*3*5, num=7500*3*5), (7500*3, 1, 5)), np.ones((7500*3, 2)))
# aaa = (np.reshape(np.linspace(0, 7500*3*5, num=7500*3*5), (7500*3*5)))
# aa = tf.data.Dataset.from_tensor_slices(aaa)
# aa = aa.batch(7500)
# bb = aa.window(size=500, shift=1, stride=1, drop_remainder=True)
# cc = bb.interleave(lambda xx, yy: tf.data.Dataset.zip((xx.map(lambda x: tf.transpose(x, [1, 0, 2])), yy)),
#                              cycle_length=3,
#                              block_length=500)
# dd = bb.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(10), y.batch(10))))
# iter = dd.make_one_shot_iterator()
# f,g = iter.get_next()
# print(iter.get_next())
#
# root_path = Path('./tf_records') / 'train'
# paths = [str(x) for x in root_path.glob('*.tfrecords')]
# paths.sort()
#
# filename_queue = tf.data.Dataset.list_files(paths,
#                                             shuffle=True,
#                                             seed=None)
#
# dataset = filename_queue.map(tf.data.TFRecordDataset)
# dataset = dataset.interleave(map_func=get_window_for_each_file_fn, cycle_length=1)
# iter = dataset.make_one_shot_iterator()
# f, g = iter.get_next()
#
# ds = tf.data.Dataset.range(7500*3)
# ds = ds.window(size=500, shift=1,
#                stride=1,
#                drop_remainder=True).flat_map(lambda x: x.batch(500))
#
# it = ds.make_one_shot_iterator()
# data = it.get_next()


def siamese_net(audio_frames=None,
                num_features=640,
                seq_length=500,
                batch_size=2,
                is_training=False,
                latent_dim=3):
    with tf.variable_scope("Siamese_net", reuse=tf.AUTO_REUSE):
        audio_input = tf.reshape(audio_frames, [-1, 1, 640, 1])

        # ### Maybe adding a batchnormalization to normalize input
        # All conv2d should be SAME padding
        # net = tf.layers.dropout(audio_input,
        #                         rate=0.5,
        #                         training=is_training,
        #                         name='Input_Dropout')
        net = tf.layers.conv2d(audio_input,
                               filters=64,
                               kernel_size=(1, 8),
                               strides=(1, 1),
                               padding='same',
                               data_format='channels_last',
                               activation=None,
                               use_bias=True,
                               name='Conv2d_1')

        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 10, 1],
            strides=[1, 1, 10, 1],
            padding='SAME',
            name='Maxpooling_1')

        net = tf.layers.dropout(net,
                                rate=0.5,
                                training=is_training,
                                name='Dropout_1')

        # Original model had 400 output filters for the second conv layer
        # but this trains much faster and achieves comparable accuracy.
        net = tf.layers.conv2d(net,
                               filters=128,
                               kernel_size=(1, 6),
                               strides=(1, 1),
                               padding='same',
                               data_format='channels_last',
                               activation=None,
                               use_bias=True,
                               name='Conv2d_2')

        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 8, 1],
            strides=[1, 1, 8, 1],
            padding='SAME',
            name='Maxpooling_2')

        net = tf.layers.dropout(net,
                                rate=0.5,
                                training=is_training,
                                name='Dropout_2')

        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(1, 6),
                               strides=(1, 1),
                               padding='same',
                               data_format='channels_last',
                               activation=None,
                               use_bias=True,
                               name='Conv2d_3')

        net = tf.reshape(net, (-1, num_features // 80, 256, 1))  # -1 -> batch_size*seq_length

        # Pooling over the feature maps.
        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 8, 1],
            strides=[1, 1, 8, 1],
            padding='SAME',
            name='Maxpooling_3')

        net = tf.layers.dropout(net,
                                rate=0.5,
                                training=is_training,
                                name='Dropout_3')

        net = tf.reshape(net, (-1, seq_length, num_features // 80 * 32))  # -1 -> batch_size

        stacked_lstm = []
        for iiLyr in range(2):
            stacked_lstm.append(
                tf.nn.rnn_cell.LSTMCell(num_units=latent_dim, use_peepholes=True, cell_clip=100, state_is_tuple=True))
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_lstm, state_is_tuple=True)

        # We have to specify the dimensionality of the Tensor so we can allocate
        # weights for the fully connected layers.
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

        net = tf.reshape(outputs, (-1, seq_length, latent_dim))  # -1 -> batch_size*seq_length

        return net


def get_label(feature_input, label, num_features, seq_length, similarity_margin=0.025):
    feature_input = tf.reshape(feature_input, (-1, 2, seq_length, num_features))
    label = tf.reshape(label, (-1, 2, seq_length, 2))

    y1 = label[:, 0, :, :]
    y2 = label[:, 1, :, :]

    spacial_dist = tf.norm(y1-y2,
                           ord='euclidean',
                           axis=-1)
    similar_pairs = spacial_dist <= similarity_margin  # -1, seq_length

    pair_label = tf.cast(similar_pairs, dtype=tf.int32)
    return feature_input, pair_label, label


def get_window_for_each_file_fn(ds, seq_length):
    ds = tf.data.TFRecordDataset(ds)
    ds = ds.map(parse_fn)
    ds = ds.window(size=seq_length, shift=2, stride=1, drop_remainder=True)
    ds = ds.interleave(lambda xx, yy: tf.data.Dataset.zip((xx.batch(seq_length), yy.batch(seq_length))),
                       cycle_length=1,
                       block_length=seq_length)
    ds = ds.map(lambda x, y: (tf.transpose(x, [1, 0, 2]), tf.squeeze(y)))
    return ds


def parse_fn(example):
  "Parse TFExample records and perform simple data augmentation."
  example_fmt = {
        'sample_id': tf.FixedLenFeature((), tf.int64),
        'subject_id': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((2), tf.float32),
        'raw_audio': tf.FixedLenFeature((640), tf.float32),
        'label_shape': tf.FixedLenFeature((), tf.int64),
        'audio_shape': tf.FixedLenFeature((), tf.int64),
    }
  features = tf.parse_single_example(example, example_fmt)
  raw_audio = features['raw_audio']
  label = features['label']

  audio_shape = features['audio_shape']
  label_shape = features['label_shape']

  raw_audio = tf.reshape(raw_audio, (-1, audio_shape))
  label = tf.reshape(label, (-1, label_shape))

  return raw_audio, label


def get_dataset(dataset_dir, is_training=True, split_name='train', batch_size=32,
              seq_length=100, debugging=False):
    """Returns a data split of the RECOLA dataset, which was saved in tfrecords format.

    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """
    root_path = Path(dataset_dir) / split_name
    paths = [str(x) for x in root_path.glob('*.tfrecords')]
    paths.sort()

    filename_queue = tf.data.Dataset.list_files(paths,
                                                shuffle=is_training,
                                                seed=None)
    if is_training:
        dataset = filename_queue.interleave(map_func=lambda x: get_window_for_each_file_fn(x, seq_length=seq_length),
                                     cycle_length=9)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size=batch_size)
    else:
        dataset = filename_queue.interleave(tf.data.TFRecordDataset, cycle_length=9)
        dataset = dataset.map(parse_fn)
        dataset = dataset.batch(batch_size=seq_length)
        dataset = dataset.map(lambda x, y: (tf.transpose(x, [1, 0, 2]), tf.squeeze(y)))
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(2)

    dataset = dataset.map(lambda x, y: get_label(x, y,
                                                 num_features=640,
                                                 seq_length=seq_length,
                                                 similarity_margin=0.05))
    dataset = dataset.prefetch(batch_size*seq_length*10)
    return dataset


def get_data_iter(output_types, output_shapes):
    # Make the iterator
    iterator = tf.data.Iterator.from_structure(output_types,
                                               output_shapes)
    dataset_iter = iterator.get_next()
    return iterator, dataset_iter


def visualizing_weights_bias():
    # Visualize all weights and bias
    var_visual_list = []
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if "kernel" or "bias" in var.name:
            var_visual_list.append(var)
    for var in var_visual_list:
        tf.summary.histogram(var.name, var)


def get_learning_rate_decay(is_decay=False,
                            global_step=None,
                            init_learning_rate=0.001,
                            epochs=None,
                            decay_type="cosine_restarts"):
    def get_decay_func(init_learning_rate=0.001,
                       global_step=None,
                       epochs=None,
                       decay_type="cosine_decay_restarts"):
        options = {"polynomial_decay":
                       tf.train.polynomial_decay(learning_rate=init_learning_rate,
                                                 global_step=global_step,
                                                 decay_steps=epochs//10,
                                                 end_learning_rate=init_learning_rate/100,
                                                 power=1.0,
                                                 cycle=False,
                                                 name="polynomial_decay_no_cycle"),
                   "inverse_time_decay":
                       tf.train.inverse_time_decay(learning_rate=init_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=epochs // 10,
                                                   decay_rate=0.95,
                                                   staircase=True,
                                                   name="inverse_time_decay_staircase"),
                   "cosine_decay_restarts":
                       tf.train.cosine_decay_restarts(learning_rate=init_learning_rate,
                                                      global_step=global_step,
                                                      first_decay_steps=epochs//4,
                                                      t_mul=2.0,
                                                      m_mul=0.5,
                                                      alpha=0.0,
                                                      name='cosine_decay_restarts'),
                   "noisy_linear_cosine_decay":
                       tf.train.noisy_linear_cosine_decay(learning_rate=init_learning_rate,
                                                          global_step=global_step,
                                                          decay_steps=epochs//10,
                                                          initial_variance=1.0,
                                                          variance_decay=0.55,
                                                          num_periods=0.5,
                                                          alpha=0.0,
                                                          beta=0.001,
                                                          name='noisy_linear_cosine_decay')}
        return options[decay_type]

    # Learning rate decay
    if is_decay is True:
        learning_rate = get_decay_func(init_learning_rate=init_learning_rate,
                                       global_step=global_step,
                                       epochs=epochs,
                                       decay_type=decay_type)
    else:
        learning_rate = init_learning_rate

    add_global = global_step.assign_add(1)
    return learning_rate, add_global


def get_optimizer_train_op(total_loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss)

    return train_op


def mean_distance_ratio(embedding_input, label, seq_length):
    label = tf.reshape(label, [-1, seq_length])
    true_labels = tf.where(tf.cast(label, dtype=tf.bool))
    false_labels = tf.where(tf.logical_not(tf.cast(label, dtype=tf.bool)))

    true_pair_distance = tf.gather(embedding_input, true_labels[:, -1], axis=1)
    false_pair_distance = tf.gather(embedding_input, false_labels[:, -1], axis=1)

    true_mean, true_mean_update_op = tf.metrics.mean(true_pair_distance, name='true_pair_distance_mean')
    false_mean, false_mean_update_op = tf.metrics.mean(false_pair_distance, name='false_pair_distance_mean')

    ratio = true_mean / false_mean

    return ratio, true_mean_update_op, false_mean_update_op


def get_metrics_op(pair_list=[], names=[], seq_length=100):
    # ACC
    with tf.name_scope('my_metrics'):
        metrics_list = []
        metrics_update_op_list = []
        for idx, item in enumerate(pair_list):
            # TODO: make this better
            if idx == 0:
                acc, acc_update_op = tf.metrics.accuracy(item[0], item[1])
                metrics_list.append(acc)
                metrics_update_op_list.append(acc_update_op)
                tf.summary.scalar('metric/{}'.format(names[idx]), acc)
            if idx == 1:
                ratio, true_mean_update_op, false_mean_update_op = mean_distance_ratio(item[0], item[1],
                                                                                       seq_length=seq_length)
                metrics_list.append(ratio)
                metrics_update_op_list.append(true_mean_update_op)
                metrics_update_op_list.append(false_mean_update_op)
                tf.summary.scalar('metric/{}'.format(names[idx]), ratio)
    return metrics_list, metrics_update_op_list


def get_metrics_init_op():
    # Metrics initializer
    metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metrics")
    metrics_vars_initializer = tf.variables_initializer(var_list=metrics_vars)
    return metrics_vars_initializer


def save_model(sess,
               output_dir,
               inputs={},
               outputs={}):
    # Save the model, can reload different checkpoint later in testing
    if os.path.exists(output_dir + '/model_files'):
        shutil.rmtree(output_dir + '/model_files')

    tf.saved_model.simple_save(sess,
                               export_dir=output_dir + '/model_files',
                               inputs=inputs,
                               outputs=outputs)


def contrastive_loss(similarity_labels, features, batch_size, seq_length, latent_dim, similarity_margin=0.05):

    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    embedding_distance = tf.norm(features[:, 0, :, :] - features[:, 1, :, :],
                                 ord='euclidean',
                                 axis=-1)
    pair_labels = similarity_labels

    mu = 1  # the margin. parameter to maybe be tuned
    distsq = tf.square(embedding_distance)
    distsq_contrastive = tf.square(tf.maximum(mu - embedding_distance, 0))

    loss = tf.reduce_mean(pair_labels * distsq + (1 - pair_labels) * distsq_contrastive)
    return loss, embedding_distance


def train(dataset_dir=None,
          init_learning_rate=0.001,
          learning_rate_decay=True,
          decay_type='cosine_decay_restarts',
          batch_size=32,
          seq_length=2,
          num_features=640,
          latent_dim=50,
          epochs=10,
          model_name='e2e_2017',
          output_dir='./output_dir'):

    total_num = 7500 * 9
    # loss_list = np.zeros((epochs, int(np.ceil(total_num/batch_size/seq_length/2))))
    # dev_loss_list = np.zeros((epochs, int(np.ceil(total_num/batch_size/seq_length/2)))) # 9 files
    g = tf.Graph()
    val_atch_size = 15
    with g.as_default():
        # Define the datasets
        train_ds = get_dataset(dataset_dir,
                             is_training=True,
                             split_name='train',
                             batch_size=batch_size,
                             seq_length=seq_length,
                             debugging=False)

        # Pay attension that the validation set should be evaluate file-wise
        dev_ds = get_dataset(dataset_dir,
                           is_training=False,
                           split_name='valid',
                           batch_size=val_atch_size,
                           seq_length=seq_length,
                           debugging=False)

        # Get tensor signature from the dataset
        iterator, dataset_iter = get_data_iter(train_ds.output_types, train_ds.output_shapes)

        # Placeholder for variable is_training and feature_input
        is_training = tf.placeholder(tf.bool, shape=())
        feature_input_1 = tf.placeholder(dtype=tf.float32,
                                       shape=[None, 1, seq_length, num_features],  # Can feed any shape
                                       name='feature_input_1_placeholder')
        feature_input_2 = tf.placeholder(dtype=tf.float32,
                                       shape=[None, 1, seq_length, num_features],  # Can feed any shape
                                       name='feature_input_2_placeholder')
        label_input = tf.placeholder(dtype=tf.float32,
                                     shape=[None, seq_length],  # Can feed any shape
                                     name='label_input_placeholder')
        origin_label_input = tf.placeholder(dtype=tf.float32,
                                            shape=[None, 2, seq_length, 2],  # Can feed any shape
                                            name='origin_label_input_placeholder')
        # Get the output tensor
        pred_1 = siamese_net(audio_frames=feature_input_1,
                             num_features=num_features,
                             seq_length=seq_length,
                             batch_size=batch_size,
                             is_training=is_training,
                             latent_dim=latent_dim)
        pred_2 = siamese_net(audio_frames=feature_input_2,
                             num_features=num_features,
                             seq_length=seq_length,
                             batch_size=batch_size,
                             is_training=is_training,
                             latent_dim=latent_dim)

        pred_1 = tf.reshape(pred_1, (-1, 1, seq_length, latent_dim))
        pred_2 = tf.reshape(pred_2, (-1, 1, seq_length, latent_dim))
        all_embeddings = tf.concat([pred_1, pred_2], axis=1)

        # Define the loss function
        total_loss, embedding_dist = contrastive_loss(similarity_labels=label_input,
                                      features=all_embeddings,
                                      batch_size=batch_size,
                                      seq_length=seq_length,
                                      latent_dim=latent_dim,
                                      similarity_margin=0.05)
        tf.summary.scalar('losses/total_loss', total_loss)

        # Visualize all weights and bias (takes a lot of space)
        visualizing_weights_bias()

        # Get learning_rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate, add_global = get_learning_rate_decay(is_decay=learning_rate_decay,
                                                            global_step=global_step,
                                                            epochs=epochs,
                                                            init_learning_rate=init_learning_rate,
                                                            decay_type=decay_type)
        tf.summary.scalar('learning_rate', learning_rate)

        # Define the optimizer
        train = get_optimizer_train_op(total_loss=total_loss, learning_rate=learning_rate)

        # MSE
        metrics_list, metrics_update_op_list = get_metrics_op(pair_list=[(label_input,
                                                                          tf.cast(embedding_dist < 0.5, dtype=tf.int32)),
                                                                         (embedding_dist, label_input)],
                                                              names=['ACC', 'mean_distance_ratio'],
                                                              seq_length=seq_length)

        # Metrics initializer
        metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metrics")
        metrics_vars_initializer = tf.variables_initializer(var_list=metrics_vars)

        # Initialize the iterator with different dataset
        training_init_op = iterator.make_initializer(train_ds)
        dev_init_op = iterator.make_initializer(dev_ds)

        with tf.Session(graph=g) as sess:
            # Define the writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(output_dir + '/log/train/', sess.graph)
            val_writer = tf.summary.FileWriter(output_dir + '/log/validation/')
            modal_saver = tf.train.Saver(max_to_keep=5)

            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            sess.run(metrics_vars_initializer)

            # Save the model, can reload different checkpoint later in testing
            save_model(sess,
                       output_dir,
                       inputs={"feature_input_1": feature_input_1,
                               "feature_input_2": feature_input_2,
                               "is_training": is_training,
                               "label_input": label_input},
                       outputs={"pred_1": pred_1,
                                "pred_2": pred_2,
                                "all_embeddings": all_embeddings})

            # Epochs
            val_old_metric, val_new_metric = [np.inf], [0]
            for epoch_no in range(epochs):
                print('\nEpoch No: {}'.format(epoch_no))
                train_loss, val_loss = 0.0, 0.0
                count_num_train, count_num_dev = 0, 0

                # Get the learning_rate if learning_rate_decay = True
                if learning_rate_decay is True:
                    _, rate = sess.run([add_global, learning_rate])
                    print('Current learning_rate: {}'.format(rate))

                # Training phase
                try:
                    sess.run(training_init_op)
                    with tqdm(total=int(total_num/batch_size/seq_length), desc='Training') as pbar:
                        while True:
                            # Retrieve the values
                            features_value, pair_label, origin_label = sess.run(dataset_iter)

                            _, loss, summary, _, = sess.run((train,
                                                            total_loss,
                                                            merged,
                                                            metrics_update_op_list),
                                                           feed_dict={feature_input_1: features_value[:, [0], :, :],
                                                                      feature_input_2: features_value[:, [1], :, :],
                                                                      is_training: True,
                                                                      label_input: pair_label,
                                                                      origin_label_input: origin_label})
                            train_loss += loss
                            # loss_list[epoch_no, count_num_train] = loss
                            pbar.update(batch_size)
                            count_num_train += 1
                except tf.errors.OutOfRangeError:
                    train_loss /= count_num_train
                    train_mse, summary = sess.run([metrics_list, merged],
                                                  feed_dict={feature_input_1: features_value[:, [0], :, :],
                                                             feature_input_2: features_value[:, [1], :, :],
                                                             is_training: False,
                                                             label_input: pair_label,
                                                             origin_label_input: origin_label})
                    train_writer.add_summary(summary, epoch_no)
                    sess.run(metrics_vars_initializer)
                    print('Training loss: {}\n'
                          'Training ACC: {}'.format(train_loss,
                                                    train_mse))


                # Validation 10 phase
                try:
                    sess.run(dev_init_op)
                    with tqdm(total=int(total_num/val_atch_size/seq_length), desc='Validation') as pbar_dev:
                        while True:
                            # Retrieve the values
                            features_value, origin_label = sess.run(dataset_iter)
                            loss, summary, _, = sess.run((total_loss,
                                                         merged,
                                                         metrics_update_op_list),
                                                        feed_dict={feature_input_1: features_value[:, [0], :, :],
                                                                   feature_input_2: features_value[:, [1], :, :],
                                                                   is_training: False,
                                                                   label_input: pair_label,
                                                                   origin_label_input: origin_label})
                            val_loss += loss
                            # dev_loss_list[epoch_no, count_num_dev] = loss
                            pbar_dev.update(int(val_atch_size))
                            count_num_dev += 1
                except tf.errors.OutOfRangeError:
                    val_loss /= count_num_dev
                    val_mse, summary = sess.run([metrics_list, merged],
                                                feed_dict={feature_input_1: features_value[:, [0], :, :],
                                                           feature_input_2: features_value[:, [1], :, :],
                                                           is_training: False,
                                                           label_input: pair_label,
                                                           origin_label_input: origin_label})
                    val_writer.add_summary(summary, epoch_no)
                    sess.run(metrics_vars_initializer)
                    print('\nEpoch: {}'.format(epoch_no))
                    print('Validation loss: {}\n'
                          'Validation 10 ACC: {}\n'.format(val_loss,
                                                           val_mse))
                    val_new_metric = [val_mse[1]]

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
            save_model(sess,
                       output_dir,
                       inputs={"feature_input_1": feature_input_1,
                               "feature_input_2": feature_input_2,
                               "is_training": is_training,
                               "label_input": label_input},
                       outputs={"pred_1": pred_1,
                                "pred_2": pred_2,
                                "all_embeddings": all_embeddings})

    return [0], [0]


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
    parser.add_argument(
        '--decay_type',
        type=str,
        default='cosine_decay_restarts',
        help="The learning decay type"
    )
    FLAGS, unparsed = parser.parse_known_args()

    output_dir = FLAGS.output_dir
    loss_list, dev_loss_list = train(FLAGS.dataset_dir,
                                     init_learning_rate=FLAGS.learning_rate,
                                     learning_rate_decay=FLAGS.learning_rate_decay,
                                     decay_type=FLAGS.decay_type,
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

