# @Time    : 24/01/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : models.py
# @Software: PyCharm

from __future__ import absolute_import, division, print_function

import tensorflow as tf


def e2e_2017(audio_frames=None,
             hidden_units=256,
             batch_size=2,
             seq_length=2,
             num_features=640,
             number_of_outputs=2,
             is_training=False):

    audio_input = tf.reshape(audio_frames, [-1, 1, 640, 1])

    # All conv2d should be SAME padding
    net = tf.layers.dropout(audio_input,
                            rate=0.5,
                            training=is_training,
                            name='Input_Dropout')
    net = tf.layers.conv2d(net,
                           filters=40,
                           kernel_size=(1, 20),
                           strides=(1, 1),
                           padding='same',
                           data_format='channels_last',
                           activation=None,
                           use_bias=True,
                           name='Conv2d_1')

    net = tf.nn.max_pool(
        net,
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 2, 1],
        padding='SAME',
        name='Maxpooling_1')

    # Original model had 400 output filters for the second conv layer
    # but this trains much faster and achieves comparable accuracy.
    net = tf.layers.conv2d(net,
                           filters=40,
                           kernel_size=(1, 40),
                           strides=(1, 1),
                           padding='same',
                           data_format='channels_last',
                           activation=None,
                           use_bias=True,
                           name='Conv2d_2')

    net = tf.reshape(net, (-1, num_features // 2, 40, 1)) # -1 -> batch_size*seq_length

    # Pooling over the feature maps.
    net = tf.nn.max_pool(
        net,
        ksize=[1, 1, 10, 1],
        strides=[1, 1, 10, 1],
        padding='SAME',
        name='Maxpooling_2')

    net = tf.reshape(net, (-1, seq_length, num_features // 2 * 4)) # -1 -> batch_size

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
    return tf.reshape(prediction, (-1, seq_length, number_of_outputs)), 0


def e2e_2018(audio_frames=None,
             hidden_units=256,
             batch_size=2,
             seq_length=2,
             num_features=640,
             number_of_outputs=2,
             is_training=False):

    # END-TO-END SPEECH EMOTION RECOGNITION USING DEEP NEURAL NETWORKS
    # Panagiotis Tzirakis, Jiehao Zhang, Bj¨orn W. Schuller
    # ICASSP 2018

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

    net = tf.reshape(net, (-1, num_features // 80, 256, 1)) # -1 -> batch_size*seq_length

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
    return tf.reshape(prediction, (-1, seq_length, number_of_outputs)), 0.0


def e2e_2018_seperateAE(audio_frames=None,
                        hidden_units=256,
                        batch_size=2,
                        seq_length=2,
                        num_features=640,
                        number_of_outputs=2,
                        is_training=False):

    # END-TO-END SPEECH EMOTION RECOGNITION USING DEEP NEURAL NETWORKS
    # Panagiotis Tzirakis, Jiehao Zhang, Bj¨orn W. Schuller
    # ICASSP 2018

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

    net = tf.reshape(net, (-1, num_features // 80, 256, 1)) # -1 -> batch_size*seq_length

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

    net_a = tf.layers.dense(net,
                            hidden_units,
                            activation=tf.nn.relu)
    net_e = tf.layers.dense(net,
                            hidden_units,
                            activation=tf.nn.relu)
    prediction_a = tf.layers.dense(net_a,
                                   1,
                                   activation=None)
    prediction_e = tf.layers.dense(net_e,
                                   1,
                                   activation=None)
    prediction = tf.concat([prediction_a, prediction_e],
                           axis=-1)  # last dimension
    loss_ae = tf.reduce_mean(tf.tensordot(tf.reshape(net_a, (-1, seq_length, hidden_units)),
                                          tf.reshape(net_e, (-1, seq_length, hidden_units)),
                                          axes=[[2], [2]]))
    tf.summary.histogram('output_activations/arousal', tf.reshape(prediction[:, 0], [-1, ]))
    tf.summary.histogram('output_activations/valence', tf.reshape(prediction[:, 1], [-1, ]))
    return tf.reshape(prediction, (-1, seq_length, number_of_outputs)), loss_ae

def e2e_2018_provide(audio_frames=None,
                     hidden_units=256,
                     batch_size=2,
                     seq_length=2,
                     num_features=640,
                     number_of_outputs=2,
                     is_training=False):
    audio_input = tf.reshape(audio_frames, [-1, num_features * seq_length, 1])
    with tf.variable_scope("audio_model"):
      net = tf.layers.conv1d(audio_input,64,8,padding = 'same', activation=tf.nn.relu)
      net = tf.layers.max_pooling1d(net,10,10)
      net = tf.layers.dropout(net,0.5,training=is_training)

      net = tf.layers.conv1d(net,128,6,padding = 'same', activation =tf.nn.relu)
      net = tf.layers.max_pooling1d(net,8,8)
      net = tf.layers.dropout(net, 0.5, training=is_training)

      net = tf.layers.conv1d(net,256,6,padding = 'same', activation =tf.nn.relu)
      net = tf.layers.max_pooling1d(net,8,8)
      net = tf.layers.dropout(net, 0.5, training=is_training)

      net = tf.reshape(net,[batch_size,seq_length,num_features//640*256]) #256])

    with tf.variable_scope("recurrent_model"):
      lstm1 = tf.contrib.rnn.LSTMCell(256,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

      lstm2 = tf.contrib.rnn.LSTMCell(256,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

      stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)

      # We have to specify the dimensionality of the Tensor so we can allocate
      # weights for the fully connected layers.
      outputs, states = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

      net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

      prediction = tf.layers.dense(net, 2)
      prediction = tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))

    return prediction, 0.0

