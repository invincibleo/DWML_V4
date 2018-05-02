#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11.12.17 10:44
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Models
# @Software: PyCharm Community Edition

from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Conv2D, Flatten, Activation, Input, Add, Lambda
from keras.layers import TimeDistributed, Reshape, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import ZeroPadding1D, MaxPooling1D, Conv1D, InputLayer
from keras.models import Model, Sequential
from keras.initializers import glorot_uniform
from keras import backend as K
import numpy as np
import keras



def SoundNet():
    """
    Builds up the SoundNet model and loads the weights from a given model file (8-layer model is kept at models/sound8.npy).
    pool size divided by 2
    :return:
    """
    model_weights = np.load('sound8.npy', encoding='latin1').item()
    model = Sequential()
    model.add(InputLayer(input_shape=(1764, 1)))

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 2, 'pool_strides': 8, 'trainable': False},

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 2, 'pool_strides': 8, 'trainable': False},

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2, 'trainable': False},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2, 'trainable': False},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 1, 'pool_strides': 4, 'trainable': False},

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2, 'trainable': False},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2, 'trainable': False},

                         # {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                         #  'kernel_size': 8, 'conv_strides': 2},
                         ]

    for x in filter_parameters:
        model.add(ZeroPadding1D(padding=x['padding']))
        model.add(Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid', trainable=x['trainable']))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']

        model.layers[-1].set_weights([weights, biases])

        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']

            # add Batchnormalization only trainable for layer 5- 7
            if x['name'] not in ['conv1', 'conv2', 'conv3', 'conv4']:
                model.add(BatchNormalization(trainable=True))
            else:
                model.add(BatchNormalization(trainable=False))
            model.layers[-1].set_weights([gamma, beta, mean, var])
            model.add(Activation('relu'))
        if 'pool_size' in x:
            model.add(MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid'))

    return model