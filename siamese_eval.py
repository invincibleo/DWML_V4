#!/usr/bin/env python
# @Time    : 14/03/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : siamese_eval.py
# @Software:

from __future__ import absolute_import, division, print_function

import argparse
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib.tensorboard.plugins import projector
# tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

def get_ds(dataset, is_training=True, batch_size=32, seq_length=100):
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=10000)
    return dataset


def plotEmbedding(emb, colordata, coordinates = [0,1] , figsize = (9,7), markersize = 4, colormap = 'jet' , title = None ):

    ''' Vizualize the embedding, and color the points according to the ground truth '''

    """    Parameters
    ------------------------
    emb :                   numpy array
                            each row is the embedding of a point

    colordata:              numpy array or list
                            length the same as the number of rows in emb - it is the groundtruth value we want to use for coloring

    coordinates:            list
                            selects the columns from emb to visualize: can visualize 2 or 3 dimensions of the embedding

    remaining params:       they are purely related to the plotting - self explanatory

    """

    # you can only pick 3 coordinates maximum - we can not 2D or 3D only
    if len(coordinates) > 3:
        coordinates = coordinates[0:3]

    fig = plt.figure(figsize = figsize)

    if len(coordinates) == 3:

        ax = fig.add_subplot(111, projection='3d')
        h = ax.scatter(emb[:,coordinates[0]], emb[:,coordinates[1]], emb[:,coordinates[2]], c=colordata, s = markersize, cmap=colormap)
        ax.set_xlabel('feature '+str(coordinates[0]+1),fontsize = 14 )
        ax.set_ylabel('feature '+str(coordinates[1]+1),fontsize = 14 )
        ax.set_zlabel('feature '+str(coordinates[2]+1),fontsize = 14 )
        ax.set_zticklabels([])

    elif len(coordinates) == 2:

        ax = fig.add_subplot(111)
        h = ax.scatter(emb[:,coordinates[0]], emb[:,coordinates[1]], c=colordata, s = markersize, cmap=colormap)
        ax.set_xlabel('feature '+str(coordinates[0]+1),fontsize = 14 )
        ax.set_ylabel('feature '+str(coordinates[1]+1),fontsize = 14 )

    #ax.set_yticklabels([])
    #ax.set_xticklabels([])
    plt.title(title, fontsize = 15)
    fig.colorbar(h)


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

    dataset = filename_queue.interleave(tf.data.TFRecordDataset, cycle_length=1)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.shuffle(buffer_size=7500*9)
    # dataset = dataset.batch(batch_size=seq_length)
    # dataset = dataset.map(lambda x, y: (dict(features=tf.transpose(x['features'], [1, 0, 2])), y))
    # dataset = dataset.repeat(100)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=20000)
    return dataset


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
        '--checkpoint_num',
        type=int,
        default=None,
        help="The checkpoint number you want to restore"
    )
    FLAGS, unparsed = parser.parse_known_args()

    output_dir = FLAGS.output_dir
    export_dir = output_dir + '/model_files'
    dataset_dir = FLAGS.dataset_dir
    total_num = 7500 * 9
    seq_length = FLAGS.seq_length
    batch_size = FLAGS.batch_size
    latent_dim = FLAGS.latent_dim
    checkpoint_num = FLAGS.checkpoint_num
    val_batch_size = batch_size
    num_rand_embeddings = 2000

    if not os.path.exists(output_dir + '/npys'):
        os.makedirs(output_dir + '/npys')

    with tf.Session(graph=tf.Graph()) as sess:
        train_ds = get_dataset(dataset_dir,
                            is_training=False,
                            split_name='train',
                            batch_size=batch_size,
                            seq_length=seq_length,
                            debugging=False)

        test_ds = get_dataset(dataset_dir,
                            is_training=False,
                            split_name='valid',
                            batch_size=batch_size,
                            seq_length=seq_length,
                            debugging=False)
        iterator = tf.data.Iterator.from_structure(test_ds.output_types,
                                                   test_ds.output_shapes)

        # Get tensor signature from the dataset
        dataset_iter = iterator.get_next()

        train_init_op = iterator.make_initializer(train_ds)
        test_init_op = iterator.make_initializer(test_ds)
        # Load the model
        model = tf.saved_model.loader.load(sess,
                                           tags=[tag_constants.SERVING],
                                           export_dir=export_dir)

        # Retrieve the tensors
        input_1_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['feature_input_1'].name
        feature_input_1 = tf.get_default_graph().get_tensor_by_name(input_1_name)
        input_2_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['feature_input_2'].name
        feature_input_2 = tf.get_default_graph().get_tensor_by_name(input_2_name)
        is_training_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['is_training'].name
        is_training = tf.get_default_graph().get_tensor_by_name(is_training_name)
        label_input_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['label_input'].name
        label_input = tf.get_default_graph().get_tensor_by_name(label_input_name)

        pred_1_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['pred_1'].name
        pred_1 = tf.get_default_graph().get_tensor_by_name(pred_1_name)
        distance_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['distance'].name
        distance = tf.get_default_graph().get_tensor_by_name(distance_name)

        # Load the checkpoint
        saver = tf.train.Saver()
        if checkpoint_num is None:
            print("CNM")
            # If None, then loading the model files will automatically load the last epoch
            # ckpt = tf.train.get_checkpoint_state(output_dir)
            # saver.restore(sess, save_path=ckpt.all_model_checkpoint_paths[-1])
        else:
            saver.restore(sess, save_path=output_dir + "/model.ckpt-" + str(checkpoint_num))

        count_num = 0
        ground_truth_all = np.zeros((total_num//batch_size, batch_size, 2))
        prediction_all = np.zeros((total_num//batch_size, batch_size, 3))
        # testing training set phase
        try:
            sess.run(train_init_op)
            with tqdm(total=int(total_num / batch_size), desc='Testing_train') as pbar:
                while True:
                    # Retrieve testing data
                    features_value, ground_truth = sess.run(dataset_iter)

                    prediction_values = sess.run(pred_1,
                                                 feed_dict={feature_input_1: features_value,
                                                            is_training: False})

                    ground_truth_all[count_num, :] = ground_truth[:, 0, :]
                    prediction_all[count_num, :, :] = prediction_values
                    pbar.update(batch_size)
                    count_num += 1
        except tf.errors.OutOfRangeError:
            ground_truth_all = np.reshape(ground_truth_all, (-1, 2))
            prediction_all = np.reshape(prediction_all, (-1, 3))

            ground_truth_all = ground_truth_all[0:total_num, :]
            prediction_all = prediction_all[0:total_num, :]
            # Save to .npy
            np.save(output_dir + '/npys/embeddings_train.npy', prediction_all)

            rand_idx = np.random.randint(prediction_all.shape[0], size=num_rand_embeddings)
            # Visualization of the embeddings
            summary_writer = tf.summary.FileWriter(output_dir + '/embeddings_train')
            embedding_var = tf.Variable(prediction_all[rand_idx, :], name='latent_z')
            sess.run(embedding_var.initializer)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = "metadata.tsv"
            projector.visualize_embeddings(summary_writer, config)
            # Write labels
            embeddings_metadata_addr = output_dir + '/embeddings_train/metadata.tsv'
            if not os.path.exists(output_dir + '/embeddings_train'):
                os.makedirs(output_dir + '/embeddings_train')
            with open(embeddings_metadata_addr, 'w') as f:
                f.write("Index\tArousal\tValence\n")
                idx = 0
                for gt in ground_truth_all[rand_idx, :]:
                    idx += 1
                    f.write("{}\t{}\t{}\n".format(idx, int(gt[0]), int(gt[1])))

            saver = tf.train.Saver()  # Save the model files into the same folder of embeddings
            saver.save(sess, os.path.join(output_dir + '/embeddings_train', "embeding_model.ckpt"), 1)

        count_num = 0
        ground_truth_all = np.zeros((total_num//batch_size, batch_size, 2))
        prediction_all = np.zeros((total_num//batch_size, batch_size, 3))
        # Testing dev_10_ds phase
        try:
            sess.run(test_init_op)
            with tqdm(total=int(total_num / seq_length), desc='Testing_10') as pbar:
                while True:
                    # Retrieve testing data
                    features_value, ground_truth = sess.run(dataset_iter)

                    prediction_values = sess.run(pred_1,
                                                 feed_dict={feature_input_1: features_value,
                                                            is_training: False})

                    ground_truth_all[count_num, :] = ground_truth[:, 0, :]
                    prediction_all[count_num, :, :] = prediction_values
                    pbar.update(batch_size)
                    count_num += 1
        except tf.errors.OutOfRangeError:
            ground_truth_all = np.reshape(ground_truth_all, (-1, 2))
            prediction_all = np.reshape(prediction_all, (-1, 3))

            ground_truth_all = ground_truth_all[0:total_num, :]
            prediction_all = prediction_all[0:total_num, :]
            # Save to .npy
            np.save(output_dir + '/npys/embeddings_valid.npy', prediction_all)

            rand_idx = np.random.randint(prediction_all.shape[0], size=num_rand_embeddings)
            # Visualization of the embeddings
            summary_writer = tf.summary.FileWriter(output_dir + '/embeddings_valid')
            embedding_var = tf.Variable(prediction_all[rand_idx, :], name='latent_z')
            sess.run(embedding_var.initializer)
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = "metadata.tsv"
            projector.visualize_embeddings(summary_writer, config)
            # Write labels
            embeddings_metadata_addr = output_dir + '/embeddings_valid/metadata.tsv'
            if not os.path.exists(output_dir + '/embeddings_valid'):
                os.makedirs(output_dir + '/embeddings_valid')
            with open(embeddings_metadata_addr, 'w') as f:
                f.write("Index\tArousal\tValence\n")
                idx = 0
                for gt in ground_truth_all[rand_idx, :]:
                    idx += 1
                    f.write("{}\t{}\t{}\n".format(idx, int(gt[0]), int(gt[1])))

            saver = tf.train.Saver()  # Save the model files into the same folder of embeddings
            saver.save(sess, os.path.join(output_dir + '/embeddings_valid', "embeding_model.ckpt"), 1)



