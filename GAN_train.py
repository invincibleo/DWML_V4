#!/usr/bin/env python
# @Time    : 18/02/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : GAN_train
# @Software: PyCharm

# Reference: https://blog.paperspace.com/implementing-gans-in-tensorflow/

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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

def generator(audio_frames=None,
                  hidden_units=256,
                  seq_length=2,
                  num_features=640,
                  latent_dim=50,
                  is_training=False,
              reuse=False):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        audio_input = tf.reshape(audio_frames, [-1, 640])
        net = tf.layers.Dense(3*num_features,
                              activation=tf.nn.relu)(audio_input)
        net = tf.layers.dropout(net, rate=0.5, training=is_training)
        net = tf.layers.Dense(3*num_features,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5, training=is_training)
        net = tf.layers.Dense(latent_dim,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5, training=is_training)
        net = tf.layers.Dense(num_features)(net)
        net = tf.reshape(net, (-1, 1, seq_length, num_features))
        return net

def discriminator(audio_frames=None,
                   hidden_units=256,
                   seq_length=2,
                   num_features=640,
                   latent_dim=50,
                   is_training=False,
                  reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        net = tf.reshape(audio_frames, (-1, num_features))
        net = tf.layers.Dense(3*num_features,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5, training=is_training)
        # net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.Dense(3*num_features,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net, rate=0.5, training=is_training)
        # net = tf.layers.batch_normalization(net, training=is_training)
        net_l = tf.layers.Dense(latent_dim,
                              activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net_l, rate=0.5, training=is_training)
        net = tf.layers.Dense(num_features)(net)
        net = tf.layers.dropout(net, rate=0.5, training=is_training)
        net = tf.layers.Dense(1)(net)
        net = tf.reshape(net, (-1, 1, seq_length, 1))
        return net, net_l


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
        noise_input = tf.placeholder(dtype=tf.float32,
                                     shape=[None, 1, seq_length, num_features],  # Can feed any shape
                                     name='noise_input_placeholder')
        G_sample = generator(noise_input)
        r_logits, r_rep = discriminator(audio_input)
        f_logits, f_rep = discriminator(G_sample, reuse=True)

        tf.summary.audio("fake_audio",
                         tf.reshape(G_sample, (batch_size, -1)),
                         sample_rate=16000,
                         max_outputs=3)

        tf.summary.histogram("fake_generation",
                             tf.reshape(G_sample, (-1, num_features)))
        tf.summary.histogram("ground_truth",
                             tf.reshape(audio_input, (-1, num_features)))

        disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(
            r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))

        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))

        tf.summary.scalar('losses/disc_loss', disc_loss)
        tf.summary.scalar('losses/gen_loss', gen_loss)
        tf.summary.scalar('losses/total_loss', disc_loss + gen_loss)
        tf.summary.histogram('losses/disc_loss', disc_loss)
        tf.summary.histogram('losses/gen_loss', gen_loss)

        # Visualize all weights and bias
        var_visual_list = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if "kernel" or "bias" in var.name:
                var_visual_list.append(var)
        for var in var_visual_list:
            tf.summary.histogram(var.name, var)

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
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

        gen_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(gen_loss, var_list=gen_vars)  # G Train step
        disc_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=disc_vars)  # D Train step

        # MSE
        with tf.name_scope('my_metrics'):
            acc, acc_update_op = tf.metrics.accuracy(labels=[tf.ones_like(r_logits), tf.zeros_like(f_logits)],
                                                     predictions=[r_logits, f_logits])
        tf.summary.scalar('metric/acc_{}'.format('over_all'), acc)

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
                            noise_input_batch = np.random.uniform(-1, 1, features_value.shape)
                            _, dloss, _, = sess.run((disc_step,
                                                    disc_loss,
                                                    acc_update_op),
                                                feed_dict={audio_input: features_value,
                                                           noise_input: noise_input_batch,
                                                           is_training: True})
                            _, gloss, summary, _ = sess.run((gen_step,
                                                 gen_loss,
                                                 merged,
                                                 acc_update_op),
                                                feed_dict={audio_input: features_value,
                                                           noise_input: noise_input_batch,
                                                           is_training: True})

                            train_loss += dloss + gloss
                            loss_list[epoch_no, count_num_train] = dloss + gloss
                            pbar.update(batch_size)
                            count_num_train += 1
                except tf.errors.OutOfRangeError:
                    train_loss /= count_num_train
                    train_acc, summary = sess.run([acc,
                                                   merged],
                                                  feed_dict={audio_input: features_value,
                                                             is_training: False})
                    sess.run(metrics_vars_initializer)
                    print('Training loss: {}\n'
                          'Training ACC: {}'.format(train_loss,
                                                    train_acc))
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
                            noise_input_batch = np.random.uniform(-1, 1, features_value.shape)
                            dloss, _ = sess.run((disc_loss,
                                                    acc_update_op),
                                                feed_dict={audio_input: features_value,
                                                           noise_input: noise_input_batch,
                                                           is_training: False})
                            summary, gloss, _ = sess.run((merged,
                                                 gen_loss,
                                                 acc_update_op),
                                                feed_dict={audio_input: features_value,
                                                           noise_input: noise_input_batch,
                                                           is_training: False})

                            val_loss += dloss + gloss
                            dev_loss_list[epoch_no, count_num_dev] = dloss + gloss
                            pbar_dev.update(int(7500/seq_length))
                            count_num_dev += 1
                except tf.errors.OutOfRangeError:
                    val_loss /= count_num_dev
                    val_acc, summary = sess.run([acc,
                                                 merged],
                                                feed_dict={audio_input: features_value,
                                                           is_training: False})
                    sess.run(metrics_vars_initializer)
                    print('\nEpoch: {}'.format(epoch_no))
                    print('Training loss: {}\n'
                          'Training ACC: {}\n'.format(train_loss,
                                                      train_acc))
                    print('Validation loss: {}\n'
                          'Validation ACC: {}\n'.format(val_loss,
                                                                val_acc))
                    val_writer.add_summary(summary, epoch_no)
                    val_new_metric = [val_acc]

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
                shutil.rmtree(output_dir + '/model_files')

            tf.saved_model.simple_save(sess,
                                       export_dir=output_dir + '/model_files',
                                       inputs={"audio_input": audio_input,
                                               "noise_input": noise_input,
                                               "is_training": is_training},
                                       outputs={"r_logits": r_logits,
                                                "r_rep": r_rep,
                                                "f_logits": f_logits,
                                                "f_rep": f_rep,
                                                "G_sample": G_sample})

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

