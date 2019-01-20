# @Time    : 18/01/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : train
# @Software: PyCharm

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import data_provider
import losses
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm

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
    return dataset.make_initializable_iterator() #.get_next() # make_one_shot_iterator

def dev_input_fn():
    dataset_dir = "./test_output_dir/tf_records"
    dataset = data_provider.get_dataset(dataset_dir,
                                        is_training=True,
                                        split_name='valid',
                                        batch_size=32,
                                        seq_length=2,
                                        debugging=False)

    # Return the read end of the pipeline.
    return dataset.make_initializable_iterator() #.get_next()  # make_one_shot_iterator

def my_model(audio_frames=None,
             audio_frames_dev=None,
             hidden_units=256,
             seq_length=2,
             num_features=640,
             number_of_outputs=2,
             is_training=False):

    # batch_size, _, seq_length, num_features = audio_frames.get_shape().as_list()
    audio_input = tf.cond(is_training,
                          lambda: tf.reshape(audio_frames, [-1, 1, 640, 1]),
                          lambda: tf.reshape(audio_frames_dev, [-1, 1, 640, 1]))# -1 -> batch_size*seq_length

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

    return tf.reshape(prediction, (-1, seq_length, number_of_outputs))

def train(dir):
    g = tf.Graph()
    with g.as_default():
        train_iter = train_input_fn()
        features, ground_truth = train_iter.get_next() #train_input_fn()
        ground_truth = tf.squeeze(ground_truth, 2)

        dev_iter = dev_input_fn()
        features_dev, ground_truth_dev = dev_iter.get_next()#dev_input_fn()
        ground_truth_dev = tf.squeeze(ground_truth_dev, 2)

        is_training = tf.placeholder(tf.bool, shape=())

        prediction = my_model(audio_frames=features['features'],
                              audio_frames_dev=features_dev['features'],
                              hidden_units=256,
                              number_of_outputs=2,
                              is_training=is_training)

        # Define the loss function
        for i, name in enumerate(['arousal', 'valence']):
            pred_single = tf.reshape(prediction[:, :, i], (-1,))
            gt_single = tf.reshape(ground_truth[:, :, i], (-1,)) # ground_truth

            loss = losses.concordance_cc(pred_single, gt_single)
            tf.Print(loss, [tf.shape(loss)], 'shape(loss) = ', first_n=2)
            # tf.summary.scalar('losses/{} loss'.format(name), loss)

            mse = tf.reduce_mean(tf.square(pred_single - gt_single))
            # tf.summary.scalar('losses/mse {} loss'.format(name), mse)

            tf.losses.add_loss(loss / 2.)

        # aa = tf.get_collection(tf.GraphKeys.LOSSES)
        total_loss = tf.losses.get_total_loss()
        # tf.summary.scalar('losses/total loss', total_loss)

        # pred_single_dev = tf.reshape(prediction_dev[:, :, 0], (-1,))
        # gt_single_dev = tf.reshape(ground_truth_dev[:, :, 0], (-1,))

        # total_loss_dev = losses.concordance_cc(pred_single_dev, gt_single_dev)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(total_loss)

        # init = tf.global_variables_initializer()
        epochs = 10
        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_no in range(epochs):
                train_loss, train_accuracy = 0, 0
                val_loss, val_accuracy = 0, 0

                # Initialize iterator with training data
                sess.run(train_iter.initializer)
                sess.run(dev_iter.initializer)

                try:
                    with tqdm(total=7500*9) as pbar:
                        while True:
                            _, loss = sess.run((train, total_loss), feed_dict={is_training: True})
                            train_loss += loss
                            # train_accuracy += acc
                            pbar.update(32*2)
                except tf.errors.OutOfRangeError:
                    print('Training loss: {}'.format(total_loss))
                    pass

                # Initialize iterator with validation data
                try:
                    while True:
                        loss = sess.run(total_loss, feed_dict={is_training: False})
                        val_loss += loss
                        # val_accuracy += acc
                except tf.errors.OutOfRangeError:
                    print('Validation loss: {}'.format(val_loss))
                    pass

                print('\nEpoch No: {}'.format(epoch_no + 1))
                # print('Train accuracy = {:.4f}, loss = {:.4f}'.format(train_accuracy / len(y_train),
                #                                                       train_loss / len(y_train)))
                # print('Val accuracy = {:.4f}, loss = {:.4f}'.format(val_accuracy / len(y_val),
                #                                                     val_loss / len(y_val)))


        # with tf.Session(graph=g) as sess:
        #     sess.run(init)
        #     for i in range(10):
        #         _, loss_value = sess.run((train, total_loss))
        #         print(loss_value)
        #
        #     print('validation loss: {}'.format(sess.run(features_dev)))

    # # Loop forever, alternating between training and validation.
    # while True:
    #     # Run 200 steps using the training dataset. Note that the training dataset is
    #     # infinite, and we resume from where we left off in the previous `while` loop
    #     # iteration.
    #     for _ in range(200):
    #         sess.run(next_element, feed_dict={handle: training_handle})
    #
    #     # Run one pass over the validation dataset.
    #     sess.run(validation_iterator.initializer)
    #     for _ in range(50):
    #         sess.run(next_element, feed_dict={handle: validation_handle})


if __name__ == "__main__":
  train(Path("./test_output_dir"))
