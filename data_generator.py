# @Time    : 15/01/2019
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : data_generator.py
# @Software: PyCharm

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import os
import librosa
import arff
import matplotlib
matplotlib.use("TkAgg") # Add only for Mac to avoid crashing
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

root_dir = Path("/Users/invincibleo/Box Sync/PhD/Experiment/Datasets/AVEC2016")

portion_to_id = dict(
    train=["train_1","train_2","train_3","train_4","train_5","train_6","train_7","train_8","train_9"],
    valid=["dev_1","dev_2","dev_3","dev_4","dev_5","dev_6","dev_7","dev_8","dev_9"],
    #test  = [54, 53, 13, 20, 22, 32, 38, 47, 48, 49, 57, 58, 59, 62, 63] # 54, 53
)

# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _str_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_list_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_list_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_list_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# Modify features here
def get_samples(subject_id):
    arousal_label_path = root_dir / 'ratings_gold_standard/arousal/{}.arff'.format(subject_id)
    valence_label_path = root_dir / 'ratings_gold_standard/valence/{}.arff'.format(subject_id)
    try:
        subsampled_audio, fs = librosa.load(str(root_dir / "recordings_audio/{}.wav".format(subject_id)), dtype='float32', sr=16000, mono=True)
        audio_frames_org = np.reshape(subsampled_audio, (-1, int(0.04 * fs)))
    except OSError as e:
        print("When reading the audio, something wrong...")
        print(e)
        return []

    audio_frames = []
    for i in range(0, audio_frames_org.shape[0]):
        audio = np.array(audio_frames_org[i, :])
        audio = audio[:int(0.04*fs)]
        audio_frames.append(audio.astype(np.float32))

    arrf_content = arff.load(open(str(arousal_label_path), 'r'))
    arousal = np.array(arrf_content['data'])
    arousal = np.array(arousal[1:, 2:], dtype='float32')  # delete the first column which is the timesteps

    arrf_content = arff.load(open(str(valence_label_path), 'r'))
    valence = np.array(arrf_content['data'])
    valence = np.array(valence[1:, 2:], dtype='float32')  # delete the first column which is the timesteps

    return audio_frames, np.dstack([arousal.T, valence.T])[0].astype(np.float32)

def serialize_sample(writer, subject_id):
    subject_name = '{}'.format(subject_id)
    for i, (audio, label) in enumerate(zip(*get_samples(subject_name))):
        label_shape = label.shape
        audio_shape = audio.shape
        feature = {
            'sample_id': _int64_feature(i),
            'subject_id': _str_feature(subject_id),
            'label': _float_list_feature(np.reshape(label, (-1,))),
            'raw_audio': _float_list_feature(np.reshape(audio, (-1,))),
            'label_shape': _int64_list_feature(label_shape),
            'audio_shape': _int64_list_feature(audio_shape)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())
        del audio, label

def main(directory):
    for portion in portion_to_id.keys():
        for subj_id in tqdm(portion_to_id[portion], desc=portion):
            if not os.path.exists(directory / 'tf_records' / portion):
                os.makedirs(directory / 'tf_records' / portion)
            writer = tf.python_io.TFRecordWriter(
                (directory / 'tf_records' / portion / '{}.tfrecords'.format(subj_id)
                ).as_posix())
            serialize_sample(writer, subj_id)

if __name__ == "__main__":
  main(Path("./test_output_dir"))


