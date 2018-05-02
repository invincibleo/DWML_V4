#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15.11.17 11:07
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Dataset_AVEC2016
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import arff
import librosa

from core.Dataset_V2 import Dataset
from core.Models import SoundNet


class Dataset_AVEC2016_V2(Dataset):
    def __init__(self, *args, **kwargs):
        dataset_name = 'Dataset_AVEC2016_V2'
        super(Dataset_AVEC2016_V2, self).__init__(self, dataset_name=dataset_name, *args, **kwargs)
        self.using_existing_features = kwargs.get('using_existing_features', True)
        self.num_classes = -1
        self.label_list = None
        self.extensions = ['wav']
        self.data_list = self.create_data_list()
        self.multi_task = ['valence_arousal', '']

    def create_data_list(self):
        datalist_pickle_file = self.get_dataset_file_addr()
        if not tf.gfile.Exists(datalist_pickle_file) or not tf.gfile.Exists(self.feature_dir):
            data_list = {'training_num': 0, 'testing_num': 0, 'validation_num': 0,
                         'validation': [], 'testing': [], 'training': [],
                         'validation_label': [], 'testing_label': [], 'training_label': [], 'label_list': []}
            if self.using_existing_features:
                feature_file_list = []
                for extension in ['arff']:
                    feature_dir = os.path.join(os.path.join(self.dataset_dir, 'features_audio'), 'arousal')
                    file_glob = os.path.join(feature_dir, '*.' + extension)
                    feature_file_list.extend(tf.gfile.Glob(file_glob))

                for existing_feature_file_addr in tqdm(feature_file_list, desc='Creating datalist:'):
                    # using existing feature files
                    feature_file = os.path.basename(existing_feature_file_addr)
                    arrf_content = arff.load(open(existing_feature_file_addr, 'r'))
                    data = np.array(arrf_content['data'], dtype=np.float16)
                    data = data[:, 1:] # delete the first column which is the timesteps
                    data_meta = np.array(arrf_content['attributes'])[1:, 0]
                    series_data = pd.DataFrame(data, columns=data_meta)
                    file_prefix = feature_file.split('.')[0]
                    self.label_list = data_meta.tolist()

                    feature_file_addr = self.get_feature_file_addr('recordings_audio', file_prefix)
                    if not tf.gfile.Exists(feature_file_addr + '.npy'):
                        feature_base_addr = '/'.join(feature_file_addr.split('/')[:-1])
                        if not tf.gfile.Exists(feature_base_addr):
                            os.makedirs(feature_base_addr)
                        save_features = True
                        features_total = []
                        labels_total = []
                    else:
                        save_features = False
                        continue

                    arousal_annotation_file_dir = os.path.join(os.path.join(self.dataset_dir, 'ratings_gold_standard'), 'arousal')
                    arousal_annotation_file_addr = arousal_annotation_file_dir + '/' + file_prefix + '.arff'
                    valence_annotation_file_dir = os.path.join(os.path.join(self.dataset_dir, 'ratings_gold_standard'), 'valence')
                    valence_annotation_file_addr = valence_annotation_file_dir + '/' + file_prefix + '.arff'

                    category = file_prefix.split('_')[0]
                    if category != 'test':
                        arousal_file_content = np.array(arff.load(open(arousal_annotation_file_addr, 'r'))['data'])
                        valence_file_content = np.array(arff.load(open(valence_annotation_file_addr, 'r'))['data'])
                        arousal_annotation = np.array(arousal_file_content[:, -1], dtype=np.float16)
                        valence_annotation = np.array(valence_file_content[:, -1], dtype=np.float16)

                        annotation = pd.DataFrame(np.array([arousal_annotation, valence_annotation]).T, columns=['arousal', 'valence'])

                        datapoint_num, feature_data, labels = self.create_windowed_datalist_with_labels(series_data=series_data,
                                                                                                        annotation=annotation,
                                                                                                        win_size=1,
                                                                                                        hop_size=1,
                                                                                                        save_features=save_features,
                                                                                                        feature_file_addr=feature_file_addr)
                        if category == 'dev':
                            data_list['validation'].append(feature_file_addr+'.npy')
                            data_list['validation_label'].append(feature_file_addr + '.labels.npy')
                            data_list['validation_num'] += datapoint_num
                            self.validation_total_features.append(feature_data)
                            self.validation_total_labels.append(labels)
                        elif category == 'train':
                            data_list['training'].append(feature_file_addr+'.npy')
                            data_list['training_label'].append(feature_file_addr + '.labels.npy')
                            data_list['training_num'] += datapoint_num
                            self.training_total_features.append(feature_data)
                            self.training_total_labels.append(labels)
                        else:
                            data_list['testing'].append(feature_file_addr+'.npy')
                            data_list['testing_label'].append(feature_file_addr + '.labels.npy')
                            data_list['testing_num'] += datapoint_num
                            self.testing_total_features.append(feature_data)
            else:
                audio_file_list = []
                for extension in self.extensions:
                    file_glob = os.path.join(self.dataset_dir + '/recordings_audio', '*.' + extension)
                    audio_file_list.extend(tf.gfile.Glob(file_glob))
                for audio_file_addr in tqdm(audio_file_list, desc='Creating features:'):
                    audio_file = os.path.basename(audio_file_addr)
                    audio_raw_all, fs = librosa.load(audio_file_addr, dtype='float16', sr=44100, mono=True)
                    audio_raw_all *= 256.0
                    audio_raw_all = pd.DataFrame(np.reshape(audio_raw_all, (1, -1)).T)
                    file_prefix = audio_file.split('.')[0]

                    feature_file_addr = self.get_feature_file_addr('recordings_audio', file_prefix)
                    if not tf.gfile.Exists(feature_file_addr + '.npy'):
                        feature_base_addr = '/'.join(feature_file_addr.split('/')[:-1])
                        if not tf.gfile.Exists(feature_base_addr):
                            os.makedirs(feature_base_addr)
                        save_features = True
                        features_total = []
                        labels_total = []
                    else:
                        save_features = False
                        continue

                    arousal_annotation_file_dir = os.path.join(os.path.join(self.dataset_dir, 'ratings_gold_standard'), 'arousal')
                    arousal_annotation_file_addr = arousal_annotation_file_dir + '/' + file_prefix + '.arff'
                    valence_annotation_file_dir = os.path.join(os.path.join(self.dataset_dir, 'ratings_gold_standard'), 'valence')
                    valence_annotation_file_addr = valence_annotation_file_dir + '/' + file_prefix + '.arff'

                    category = file_prefix.split('_')[0]
                    if category != 'test':
                        arousal_file_content = np.array(arff.load(open(arousal_annotation_file_addr, 'r'))['data'])
                        valence_file_content = np.array(arff.load(open(valence_annotation_file_addr, 'r'))['data'])
                        arousal_annotation = np.array(arousal_file_content[:, -1], dtype=np.float16)
                        valence_annotation = np.array(valence_file_content[:, -1], dtype=np.float16)

                        annotation = pd.DataFrame(np.array([arousal_annotation, valence_annotation]).T, columns=['arousal', 'valence'])
                        datapoint_num, feature_data, labels = self.create_windowed_datalist_with_labels(series_data=audio_raw_all,
                                                                                                        annotation=annotation,
                                                                                                        win_size=self.FLAGS.fs*0.04*int(self.FLAGS.dimension.split(',')[0]),
                                                                                                        hop_size=self.FLAGS.fs*0.04*1,
                                                                                                        # feature_delay_list=np.linspace(0, 8.0, 21),
                                                                                                        save_features=save_features,
                                                                                                        feature_file_addr=feature_file_addr)

                        if category == 'dev':
                            data_list['validation'].append(feature_file_addr+'.npy')
                            data_list['validation_label'].append(feature_file_addr + '.labels.npy')
                            data_list['validation_num'] += datapoint_num
                            self.validation_total_features.append(feature_data)
                            self.validation_total_labels.append(labels)
                        elif category == 'train':
                            data_list['training'].append(feature_file_addr+'.npy')
                            data_list['training_label'].append(feature_file_addr + '.labels.npy')
                            data_list['training_num'] += datapoint_num
                            self.training_total_features.append(feature_data)
                            self.training_total_labels.append(labels)
                        else:
                            data_list['testing'].append(feature_file_addr+'.npy')
                            data_list['testing_label'].append(feature_file_addr + '.labels.npy')
                            data_list['testing_num'] += datapoint_num
                            self.testing_total_features.append(feature_data)

            data_list['label_list'] = self.label_list
            pickle.dump(data_list, open(datalist_pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(datalist_pickle_file, 'rb'))
            for addr in data_list['training']:
                feature_data = np.load(addr)
                self.training_total_features.append(feature_data)
            for addr in data_list['validation']:
                feature_data = np.load(addr)
                self.validation_total_features.append(feature_data)
            for addr in data_list['training_label']:
                labels = np.load(addr)
                self.training_total_labels.append(labels)
            for addr in data_list['validation_label']:
                labels = np.load(addr)
                self.validation_total_labels.append(labels)

        # count data point
        self.num_training_data = data_list['training_num']
        self.num_validation_data = data_list['validation_num']
        self.num_testing_data = data_list['testing_num']

        self.label_list = data_list['label_list']

        self.training_total_features = np.concatenate(self.training_total_features, axis=0)
        self.training_total_labels = np.concatenate(self.training_total_labels, axis=0)

        self.validation_total_features = np.concatenate(self.validation_total_features, axis=0)
        self.validation_total_labels = np.concatenate(self.validation_total_labels, axis=0)

        # chose which delay labels to use
        # j = int(self.FLAGS.delay / 0.04 / 10)
        # self.training_total_labels = self.validation_total_labels[:, j, :]
        # self.validation_total_labels = self.validation_total_labels[:, j, :]

        # normalization
        self.training_mean = np.mean(self.training_total_features, axis=0, keepdims=True)
        self.training_std = np.std(self.training_total_features, axis=0, keepdims=True) + np.finfo(np.float16).eps

        self.training_total_features = (self.training_total_features - self.training_mean) / self.training_std
        self.validation_total_features = (self.validation_total_features - self.training_mean) / self.training_std
        return data_list

    def create_windowed_datalist_with_labels(self, series_data, annotation, win_size, hop_size, save_features, feature_file_addr):
        if hop_size == 0 or hop_size > win_size:
            raise Exception('hop_size must in range [1,win_size]! \n')
        first_element_idx_list = np.arange(0, np.shape(series_data)[0] - win_size + 1, hop_size, dtype='int32')
        # initialize features and labels to zeros
        feature_data = np.zeros((first_element_idx_list.shape[0], int(win_size), 1))
        labels = np.zeros((first_element_idx_list.shape[0], 2))

        last_element_idx_list = (first_element_idx_list + win_size).astype('int32')
        # put features and annotation in the correct locations
        for i, first_element_idx in enumerate(first_element_idx_list):
            last_element_idx = last_element_idx_list[i]
            data_point = series_data.iloc[first_element_idx: last_element_idx]
            data_point = np.expand_dims(data_point, axis=0)
            feature_data[i, :, :] = data_point

            if self.FLAGS.delay != 0.0:
                annotation_delayed_buf = annotation.append([annotation.iloc[-1, :]]*int(np.floor(self.FLAGS.delay / 0.04)), ignore_index=True)
            else:
                annotation_delayed_buf = annotation
            labels[i, :] = np.expand_dims(annotation_delayed_buf.iloc[(np.floor(last_element_idx / 1764) + np.floor(self.FLAGS.delay / 0.04)).astype('int32'), :], axis=0)

        datapoint_num = np.shape(feature_data)[0]

        time_span = int(win_size / (44100 * 0.04))
        length = int(feature_data.shape[1] / time_span)
        width = feature_data.shape[2]
        feature_data = np.reshape(feature_data, (-1, length, width))
        feature_data = self.get_SoundNet_features(feature_data)
        feature_data = np.reshape(feature_data, (datapoint_num, time_span, -1))

        if save_features:
            self.save_features_to_file(feature_data, labels, feature_file_addr)
        return datapoint_num, feature_data, labels

    def get_SoundNet_features(self, featuers):
        model = SoundNet()
        features_after = model.predict(featuers, batch_size=1024, verbose=0)
        return features_after