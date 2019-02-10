# @Time    : 08/02/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : VAE_train.py
# @Software: PyCharm
# Reference: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
# De-convolution: https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0
# Eager_implementation: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb

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

        # net = tf.reshape(net, (-1, seq_length, num_features // 80 * 32)) # -1 -> batch_size
        # stacked_lstm = []
        # for iiLyr in range(2):
        #     stacked_lstm.append(
        #         tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, use_peepholes=True, cell_clip=100, state_is_tuple=True))
        # stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_lstm, state_is_tuple=True)
        #
        # # We have to specify the dimensionality of the Tensor so we can allocate
        # # weights for the fully connected layers.
        # outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)
        #
        # net = tf.reshape(outputs, (-1, hidden_units)) # -1 -> batch_size*seq_length

        net = tf.reshape(net, (-1, seq_length, num_features // 80 * 32))
        net = tf.keras.layers.Dense(latent_dim + latent_dim)(net)
        net = tf.reshape(net, (-1, seq_length, latent_dim + latent_dim)) # -1 -> batch_size

        return net

def generative_net(audio_frames=None,
                   hidden_units=256,
                   seq_length=2,
                   num_features=640,
                   latent_dim=50,
                   is_training=False):

    with tf.variable_scope("Decoder"):
        latent_input = tf.reshape(audio_frames, [-1, latent_dim])
        net = tf.keras.layers.Dense(units=num_features//80*32, activation=tf.nn.relu)(latent_input)
        # net = tf.layers.dropout(net,
        #                         rate=0.5,
        #                         training=is_training,
        #                         name='de_Dropout_3')
        #
        # net = tf.reshape(net, (-1, seq_length, num_features // 80 * 32))
        #
        # stacked_lstm_de = []
        # for iiLyr in range(2):
        #     stacked_lstm_de.append(
        #         tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, use_peepholes=True, cell_clip=100, state_is_tuple=True))
        # stacked_lstm_de = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_lstm_de, state_is_tuple=True)
        # outputs_de, _ = tf.nn.dynamic_rnn(stacked_lstm_de, net, dtype=tf.float32)

        # net = tf.reshape(outputs_de, (-1, hidden_units))
        # net = tf.keras.layers.Dense(units=num_features // 80 * 32, activation=tf.nn.relu)(net)
        net = tf.layers.dropout(net,
                                rate=0.5,
                                training=is_training,
                                name='de_Dropout_2')

        net = tf.reshape(net, [-1, num_features//80, 32, 1])
        net = tf.image.resize_images(images=net,
                                     size=[num_features//80, 32*8],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = tf.reshape(net, [-1, 1, num_features//80, 256])
        net = tf.keras.layers.Conv2DTranspose(filters=256,
                                              kernel_size=(1, 6),
                                              strides=(1, 1),
                                              padding="same",
                                              activation=None,
                                              use_bias=True)(net)
        net = tf.image.resize_images(images=net,
                                     size=[1, num_features//80*8],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = tf.layers.dropout(net,
                                rate=0.5,
                                training=is_training,
                                name='de_Dropout_1')
        net = tf.keras.layers.Conv2DTranspose(filters=128,
                                              kernel_size=(1, 6),
                                              strides=(1, 1),
                                              padding="same",
                                              activation=None,
                                              use_bias=True)(net)
        net = tf.image.resize_images(images=net,
                                     size=[1, num_features//80*8*10],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = tf.layers.dropout(net,
                                rate=0.5,
                                training=is_training,
                                name='de_Dropout_0')
        net = tf.keras.layers.Conv2DTranspose(filters=64,
                                              kernel_size=(1, 8),
                                              strides=(1, 1),
                                              padding="same",
                                              activation=None,
                                              use_bias=True)(net)
        net = tf.keras.layers.Conv2DTranspose(filters=1,
                                              kernel_size=(1, 8),
                                              strides=(1, 1),
                                              padding="same",
                                              activation=None,
                                              use_bias=True)(net)
        net = tf.reshape(net, [-1, 1, seq_length, num_features])
        return net


def train(dataset_dir=None,
          init_learning_rate=0.001,
          learning_rate_decay=True,
          batch_size=32,
          seq_length=2,
          num_features=640,
          epochs=10,
          model_name='e2e_2017',
          output_dir='./output_dir'):

    total_num = 7500 * 9
    latent_dim = 50
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

        mean, logvar = tf.split(z, num_or_size_splits=2, axis=2)
        tf.summary.histogram("latent_mean", mean)
        tf.summary.histogram("latent_logvar", logvar)
        eps = tf.random_normal(shape=tf.shape(mean))
        z_reparameterized = eps * tf.exp(logvar * .5) + mean
        x_logit = generative_net(audio_frames=z_reparameterized,
                                 seq_length=seq_length,
                                 num_features=num_features,
                                 latent_dim=latent_dim,
                                 is_training=is_training)
        apply_sigmoid = False
        if apply_sigmoid:
            x_logit = tf.sigmoid(x_logit)

        tf.summary.audio("reconstruction_audio",
                         tf.reshape(x_logit, (batch_size, -1)),
                         sample_rate=16000,
                         max_outputs=5)
        tf.summary.histogram("reconstruction",
                             tf.reshape(x_logit, (-1, num_features)))
        tf.summary.histogram("ground_truth",
                             tf.reshape(audio_input, (-1, num_features)))

        def log_normal_pdf(sample, mean, logvar, raxis=[1, 2]):
            # sample = tf.reshape(sample, (-1, latent_dim))
            log2pi = tf.log(2. * np.pi)
            log_pdf = tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                                    axis=raxis)
            # log_pdf = tf.reshape(log_pdf, (-1, seq_length, latent_dim))
            # return tf.reduce_sum(log_pdf, axis=1)
            return log_pdf

        # x_reshaped = tf.reshape(audio_input, [-1,  num_features])
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=audio_input)
        # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpx_z = -tf.losses.mean_squared_error(predictions=x_logit, labels=audio_input)
        logpz = log_normal_pdf(z_reparameterized, 0., 0.)
        logqz_x = log_normal_pdf(z_reparameterized, mean, logvar)
        total_loss = -tf.reduce_mean(logpz - logqz_x) + logpx_z

        tf.summary.histogram('losses/logpx_z', logpx_z)
        tf.summary.histogram('losses/logpz', logpz)
        tf.summary.histogram('losses/logqz_x', logqz_x)
        tf.summary.scalar('losses/total_loss', total_loss)
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
            modal_saver = tf.train.Saver(max_to_keep=10,
                                         keep_checkpoint_every_n_hours=1)

            # Initialize the variables
            sess.run(tf.global_variables_initializer())
            sess.run(metrics_vars_initializer)

            # Epochs
            val_old_metric, val_new_metric = [0.0, 0.0], [0.0, 0.0]
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

                            _, loss, summary, _ = sess.run((train,
                                                            total_loss,
                                                            merged,
                                                            mse_update_op),
                                                           feed_dict={audio_input: features_value,
                                                                      is_training: True})
                            train_loss += loss
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
                            loss, summary, _ = sess.run((total_loss,
                                                         merged,
                                                         mse_update_op),
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
                if val_new_metric >= [x*0.9 for x in val_old_metric]:
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
                                                "z_parameterized": z_reparameterized,
                                                "x_logit": x_logit,
                                                "mean": mean,
                                                "logvar": logvar})

    return loss_list, dev_loss_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    loss_list, dev_loss_list = train(Path("./tf_records"),
                                     init_learning_rate=FLAGS.learning_rate,
                                     learning_rate_decay=FLAGS.learning_rate_decay,
                                     seq_length=FLAGS.seq_length,
                                     batch_size=FLAGS.batch_size,
                                     num_features=640,
                                     epochs=FLAGS.epochs,
                                     model_name=FLAGS.model,
                                     output_dir=output_dir)
    print(str(loss_list))
    print('\n')
    print(str(dev_loss_list))

    # Save the results
    np.savetxt(output_dir + "/loss_list.txt", loss_list, delimiter=',')
    np.savetxt(output_dir + "/dev_loss_list.txt", dev_loss_list, delimiter=',')

