# @Time    : 24/01/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : models.py
# @Software: PyCharm

from __future__ import absolute_import, division, print_function

import tensorflow as tf


def e2e_2017(audio_frames=None,
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
    tf.summary.histogram('output_activations/arousal', tf.reshape(prediction[:, 0], [-1, ]))
    tf.summary.histogram('output_activations/valence', tf.reshape(prediction[:, 1], [-1, ]))
    return tf.reshape(prediction, (-1, seq_length, number_of_outputs))

