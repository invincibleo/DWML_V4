# @Time    : 18/01/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : train
# @Software: PyCharm

# Reference and tutorial
# How to do training and validation alternatively https://zhuanlan.zhihu.com/p/43356309

from __future__ import absolute_import, division, print_function

import sys
import tensorflow as tf
import data_provider
import losses
import metrics
import numpy as np
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


def input_fn(dataset_dir, is_training=True, split_name=None, batch_size=32, seq_length=2):
    dataset = data_provider.get_dataset(dataset_dir,
                                        is_training=is_training,
                                        split_name=split_name,
                                        batch_size=batch_size,
                                        seq_length=seq_length,
                                        debugging=False)

    # Return the read end of the pipeline.
    return dataset #.make_initializable_iterator() #.get_next() # make_one_shot_iterator


def my_model(audio_frames=None,
             hidden_units=256,
             seq_length=2,
             num_features=640,
             number_of_outputs=2,
             is_training=False):

    audio_input = tf.reshape(audio_frames['features'], [-1, 1, 640, 1])

    # All conv2d should be SAME padding
    net = tf.layers.dropout(audio_input,
                            rate=0.5,
                            training=is_training,
                            name='Input_Dropout')
    net = tf.layers.conv2d(net,
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

    net = tf.layers.conv2d(net,
                           filters=256,
                           kernel_size=(1, 6),
                           strides=(1, 1),
                           padding='same',
                           data_format='channels_last',
                           activation=None,
                           use_bias=True,
                           name='Conv2d_3')

    net = tf.reshape(net, (-1, num_features // 80, 256, 1)) # -1 -> batch_size*seq_length

    # Pooling over the feature maps.
    net = tf.nn.max_pool(
        net,
        ksize=[1, 1, 8, 1],
        strides=[1, 1, 8, 1],
        padding='SAME',
        name='Maxpooling_3')

    net = tf.reshape(net, (-1, seq_length, num_features // 80 * 32)) # -1 -> batch_size

    stacked_lstm = []
    for iiLyr in range(2):
        stacked_lstm.append(
            tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, use_peepholes=True, cell_clip=100, state_is_tuple=True))
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_lstm, state_is_tuple=True)

    # We have to specify the dimensionality of the Tensor so we can allocate
    # weights for the fully connected layers.
    outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

    net = tf.reshape(outputs, (-1, hidden_units)) # -1 -> batch_size*seq_length

    prediction = tf.layers.dense(net,
                                 number_of_outputs,
                                 activation=None)
    tf.summary.histogram('activations', prediction)
    return tf.reshape(prediction, (-1, seq_length, number_of_outputs))


def train(dataset_dir=None,
          learning_rate=0.001,
          batch_size=32,
          seq_length=2,
          num_features=640,
          epochs=10,
          ):
    total_num = 7500 * 9
    loss_list = np.zeros((epochs, int(np.ceil(total_num/seq_length/batch_size))))
    dev_loss_list = np.zeros((epochs, 9)) # 9 files

    g = tf.Graph()
    with g.as_default():
        # Define the datasets
        train_ds = input_fn(dataset_dir,
                            is_training=True,
                            batch_size=batch_size,
                            seq_length=seq_length,
                            split_name='train')

        # Pay attension that the validation set should be evaluate file-wise
        dev_ds = input_fn(dataset_dir,
                          is_training=False,
                          batch_size=int(7500/seq_length),
                          seq_length=seq_length,
                          split_name='valid')

        # Make the iterator
        iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                                   dev_ds.output_shapes)

        is_training = tf.placeholder(tf.bool, shape=())

        features, ground_truth = iterator.get_next() #train_input_fn()
        ground_truth = tf.squeeze(ground_truth, 2)

        prediction = my_model(audio_frames=features,
                              hidden_units=256,
                              seq_length=seq_length,
                              num_features=num_features,
                              number_of_outputs=2,
                              is_training=is_training)

        # Define the loss function
        concordance_cc2_list = []
        names_to_updates_list = []
        mse_list = []
        mse_update_op_list = []
        for i, name in enumerate(['arousal', 'valence']):
            pred_single = tf.reshape(prediction[:, :, i], (-1,))
            gt_single = tf.reshape(ground_truth[:, :, i], (-1,))  # ground_truth

            loss = losses.concordance_cc(pred_single, gt_single)
            tf.summary.scalar('{}losses/{} loss'.format(tf.cond(is_training, lambda: 'train', lambda: 'valid'), name), loss)

            tf.losses.add_loss(loss / 2.)

            # Define some metrics
            # CCC
            concordance_cc2, _, names_to_updates = metrics.concordance_cc2(pred_single, gt_single)
            concordance_cc2_list.append(concordance_cc2)
            names_to_updates_list.append(names_to_updates)
            # MSE
            mse, mse_update_op = tf.metrics.mean_squared_error(gt_single, prediction)
            mse_list.append(mse)
            mse_update_op_list.append(mse_update_op)
            # tf.summary.scalar('losses/mse {} loss'.format(name), mse)

        # aa = tf.get_collection(tf.GraphKeys.LOSSES)
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(total_loss)

        metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metrics")
        metrics_vars_initializer = tf.variables_initializer(var_list=metrics_vars)

        with tf.Session(graph=g) as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('./log', sess.graph)
            modal_saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            sess.run(metrics_vars_initializer)
            for epoch_no in range(epochs):
                print('\nEpoch No: {}'.format(epoch_no))
                train_loss, val_loss = 0.0, 0.0
                count_num_train, count_num_dev = 0, 0

                # Initialize the iterator with different dataset
                training_init_op = iterator.make_initializer(train_ds)
                dev_init_op = iterator.make_initializer(dev_ds)
                try:
                    sess.run(training_init_op)
                    with tqdm(total=int(total_num/seq_length), desc='Training') as pbar:
                        while True:
                            _, loss, summary, _, _ = sess.run((train,
                                                               total_loss,
                                                               merged,
                                                               names_to_updates_list[0],
                                                               names_to_updates_list[1],
                                                               mse_update_op_list),
                                                              feed_dict={is_training: True})
                            train_loss += loss
                            loss_list[epoch_no, count_num_train] = loss
                            pbar.update(batch_size)
                            count_num_train += 1
                except tf.errors.OutOfRangeError:
                    train_loss /= count_num_train
                    train_ccc_arousal = sess.run(concordance_cc2_list[0])
                    train_ccc_valence = sess.run(concordance_cc2_list[1])
                    train_mse_arousal = sess.run(mse_list[0])
                    train_mse_valence = sess.run(mse_list[1])
                    sess.run(metrics_vars_initializer)
                    print('Training loss: {}\n'
                          'Training arousal CCC: {}\n'
                          'Training valence CCC: {}\n'
                          'Training arousal MSE: {}\n'
                          'Training valence MSE: {}'.format(train_loss,
                                                            train_ccc_arousal,
                                                            train_ccc_valence,
                                                            train_mse_arousal,
                                                            train_mse_valence))
                    train_writer.add_summary(summary, epoch_no)

                # Validation phase
                try:
                    sess.run(dev_init_op)
                    with tqdm(total=int(total_num/seq_length), desc='Validation') as pbar_dev:
                        while True:
                            loss, summary, _, _ = sess.run((total_loss,
                                                            merged,
                                                            names_to_updates_list[0],
                                                            names_to_updates_list[1],
                                                            mse_update_op_list),
                                                           feed_dict={is_training: False})
                            val_loss += loss
                            dev_loss_list[epoch_no, count_num_dev] = loss
                            pbar_dev.update(int(7500/seq_length))
                            count_num_dev += 1
                except tf.errors.OutOfRangeError:
                    val_loss /= count_num_dev
                    val_ccc_arousal = sess.run(concordance_cc2_list[0])
                    val_ccc_valence = sess.run(concordance_cc2_list[1])
                    val_mse_arousal = sess.run(mse_list[0])
                    val_mse_valence = sess.run(mse_list[1])
                    sess.run(metrics_vars_initializer)
                    print('\nEpoch: {}'.format(epoch_no))
                    print('Training loss: {}, '
                          'Training arousal CCC: {}, '
                          'Training valence CCC: {}, '
                          'Training arousal MSE: {}, '
                          'Training valence MSE: {}'.format(train_loss,
                                                            train_ccc_arousal,
                                                            train_ccc_valence,
                                                            train_mse_arousal,
                                                            train_mse_valence))
                    print('Validation loss: {},'
                          'Validation arousal CCC: {}, '
                          'Validation valence CCC: {}, '
                          'Validation arousal MSE: {}, '
                          'Validation valence MSE: {}'.format(val_loss,
                                                              val_ccc_arousal,
                                                              val_ccc_valence,
                                                              val_mse_arousal,
                                                              val_mse_valence))
                    train_writer.add_summary(summary, epoch_no)

            # Save the model
            save_path = modal_saver.save(sess, "./test_output_dir/model.ckpt")
            print("Model saved in path: %s" % save_path)
    return loss_list, dev_loss_list


if __name__ == "__main__":
    loss_list, dev_loss_list = train(Path("./test_output_dir/tf_records"),
                                     learning_rate=0.0001,
                                     seq_length=150,
                                     batch_size=25,
                                     num_features=640,
                                     epochs=20)
    print(str(loss_list))
    print('\n')
    print(str(dev_loss_list))

    # Save the results
    np.savetxt("./test_output_dir/loss_list.txt", loss_list, delimiter=',')
    np.savetxt("./test_output_dir/dev_loss_list.txt", dev_loss_list, delimiter=',')

