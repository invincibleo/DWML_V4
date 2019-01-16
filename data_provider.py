# @Time    : 15/01/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : data_provider.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

tf.enable_eager_execution()
slim = tf.contrib.slim

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

def get_split(dataset_dir, is_training=True, split_name='train', batch_size=32,
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
    dataset = dataset.batch(batch_size=seq_length)
    dataset = dataset.map(lambda x, y: (tf.transpose(x, [1, 0, 2]), y))
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=20000/seq_length/batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    return dataset

# def main(dataset_dir):
#     get_split(dataset_dir, is_training=True, split_name='train',
#               batch_size=32, seq_length=100, debugging=False)
#
# if __name__ == "__main__":
#   main(Path("./test_output_dir/tf_records"))
