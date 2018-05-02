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
import csv
import pandas
import hashlib
import arff

from core.Dataset import Dataset
from core.GeneralFileAccessor import GeneralFileAccessor
from core.TimeseriesPoint import *
from core.Preprocessing import *
from core.util import *
from core.ontologyProcessing import *


class Dataset_AVEC2016(Dataset):
    def __init__(self, *args, **kwargs):
        dataset_name = 'Dataset_AVEC2016'
        super(Dataset_AVEC2016, self).__init__(self, dataset_name=dataset_name, *args, **kwargs)
        self.aso = {}
        self.if_second_level_labels = kwargs.get('if_second_level_labels', False)
        self.using_existing_features = kwargs.get('using_existing_features', True)
        self.data_list_meta = []
        self.label_list = None
        self.num_classes = -1
        self.extensions = ['wav']
        self.data_list = self.create_data_list()
        self.multi_task = ['valence_arousal', '']

        if self.normalization:
            self.dataset_normalization()

        self.count_sets_data()

    def create_data_list(self):
        datalist_pickle_file = self.get_dataset_file_addr()
        if not tf.gfile.Exists(datalist_pickle_file) or not tf.gfile.Exists(self.feature_dir):
            data_list = {'validation': [], 'testing': [], 'training': []}
            if self.using_existing_features:
                feature_file_list = []
                for extension in ['arff']:
                    feature_dir = os.path.join(os.path.join(self.dataset_dir, 'features_audio'), 'arousal')
                    file_glob = os.path.join(feature_dir, '*.' + extension)
                    feature_file_list.extend(tf.gfile.Glob(file_glob))

                for existing_feature_file_addr in tqdm(feature_file_list, desc='Creating datalist:'):
                    feature_file = os.path.basename(existing_feature_file_addr)
                    feature_data = numpy.array(arff.load(open(existing_feature_file_addr, 'rb'))['data'])[:, 1:]  # the first dim is frame time
                    num_points = len(feature_data) - 1 #the last point is the replica of the second last

                    file_prefix = feature_file.split('.')[0]

                    feature_file_addr = self.get_feature_file_addr('recordings_audio', file_prefix)
                    if not tf.gfile.Exists(feature_file_addr):
                        feature_base_addr = '/'.join(feature_file_addr.split('/')[:-1])
                        if not tf.gfile.Exists(feature_base_addr):
                            os.makedirs(feature_base_addr)
                        save_features = True
                        features = {}
                    else:
                        save_features = False

                    arousal_annotation_file_dir = os.path.join(os.path.join(self.dataset_dir, 'ratings_gold_standard'), 'arousal')
                    arousal_annotation_file_addr = arousal_annotation_file_dir + '/' + file_prefix + '.arff'
                    valence_annotation_file_dir = os.path.join(os.path.join(self.dataset_dir, 'ratings_gold_standard'), 'valence')
                    valence_annotation_file_addr = valence_annotation_file_dir + '/' + file_prefix + '.arff'

                    category = file_prefix.split('_')[0]
                    if category != 'test':
                        arousal_annotation = np.array(arff.load(open(arousal_annotation_file_addr, 'rb'))['data'])
                        valence_annotation = np.array(arff.load(open(valence_annotation_file_addr, 'rb'))['data'])
                        time_index = np.asfarray(arousal_annotation[:, 1])
                        arousal_annotation = np.asfarray(arousal_annotation[:, -1]) # (np.asfarray(arousal_annotation[:, -1]) + 1) / 2    # make it between 0,1
                        valence_annotation = np.asfarray(valence_annotation[:, -1]) # (np.asfarray(valence_annotation[:, -1]) + 1) / 2    # make it between 0,1

                    if self.FLAGS.coding == 'number':
                        label_name = ['arousal', 'valence']
                        for point_idx in range(num_points):
                            start_time = time_index[point_idx]
                            end_time = time_index[point_idx + 1]

                            if category != 'test':
                                label_content = np.zeros((1, 2))
                                label_content[0, 0] = arousal_annotation[point_idx]
                                label_content[0, 1] = valence_annotation[point_idx]
                            else:
                                label_content = None
                            new_point = AudioPoint(
                                data_name=file_prefix + '.wav',
                                sub_dir='recordings_audio',
                                label_name=label_name,
                                label_content=label_content,
                                extension='wav',
                                fs=44100,
                                feature_idx=point_idx,
                                start_time=start_time,
                                end_time=end_time
                            )

                            if category == 'dev':
                                data_list['validation'].append(new_point)
                            elif category == 'test':
                                data_list['testing'].append(new_point)
                            else:
                                data_list['training'].append(new_point)

                            if save_features:
                                # feature extraction
                                features[point_idx] = np.reshape(feature_data[point_idx], (-1,))

                    if save_features:
                        self.save_features_to_file(features, feature_file_addr)
                        # pickle.dump(features, open(feature_file_addr, 'wb'), 2)
            else:
                audio_file_list = []
                for extension in self.extensions:
                    file_glob = os.path.join(self.dataset_dir + '/recordings_audio', '*.' + extension)
                    audio_file_list.extend(tf.gfile.Glob(file_glob))
                for audio_file_addr in tqdm(audio_file_list, desc='Creating features:'):
                    audio_file = os.path.basename(audio_file_addr)
                    audio_raw_all, fs = GeneralFileAccessor(file_path=audio_file_addr,
                                                            mono=True).read()
                    num_points = int(np.floor(len(audio_raw_all) / (fs * 0.04)))

                    feature_file_addr = self.get_feature_file_addr('', audio_file)
                    if not tf.gfile.Exists(feature_file_addr):
                        feature_base_addr = '/'.join(feature_file_addr.split('/')[:-1])
                        if not tf.gfile.Exists(feature_base_addr):
                            os.makedirs(feature_base_addr)
                        save_features = True
                        features = {}
                    else:
                        save_features = False

                    file_prefix = audio_file.split('.')[0]
                    arousal_annotation_file_dir = os.path.join(os.path.join(self.dataset_dir, 'ratings_gold_standard'), 'arousal')
                    arousal_annotation_file_addr = arousal_annotation_file_dir + '/' + file_prefix + '.arff'
                    valence_annotation_file_dir = os.path.join(os.path.join(self.dataset_dir, 'ratings_gold_standard'), 'valence')
                    valence_annotation_file_addr = valence_annotation_file_dir + '/' + file_prefix + '.arff'

                    category = file_prefix.split('_')[0]
                    if category != 'test':
                        arousal_annotation = np.array(arff.load(open(arousal_annotation_file_addr, 'rb'))['data'])
                        valence_annotation = np.array(arff.load(open(valence_annotation_file_addr, 'rb'))['data'])
                        time_index = np.asfarray(arousal_annotation[:, 1])
                        arousal_annotation = np.asfarray(arousal_annotation[:, -1]) # (np.asfarray(arousal_annotation[:, -1]) + 1) / 2    # make it between 0,1
                        valence_annotation = np.asfarray(valence_annotation[:, -1]) # (np.asfarray(valence_annotation[:, -1]) + 1) / 2    # make it between 0,1


                    if self.FLAGS.coding == 'number':
                        label_name = ['arousal', 'valence']
                        for point_idx in range(num_points):
                            if category != 'test':
                                label_content = np.zeros((1, 2))
                                start_time = time_index[point_idx]
                                end_time = time_index[point_idx + 1]
                                label_content[0, 0] = arousal_annotation[point_idx]
                                label_content[0, 1] = valence_annotation[point_idx]
                            else:
                                label_content = None
                                start_time = point_idx * self.FLAGS.time_resolution
                                end_time = (point_idx + 1) * self.FLAGS.time_resolution

                            new_point = AudioPoint(
                                data_name=file_prefix + '.wav',
                                sub_dir='',
                                label_name=label_name,
                                label_content=label_content,
                                extension='wav',
                                fs=44100,
                                feature_idx=point_idx,
                                start_time=start_time,
                                end_time=end_time
                            )

                            if category == 'dev':
                                data_list['validation'].append(new_point)
                            elif category == 'test':
                                data_list['testing'].append(new_point)
                            else:
                                data_list['training'].append(new_point)

                            if save_features:
                                # feature extraction
                                audio_raw = audio_raw_all[int(math.floor(start_time * fs)):int(math.ceil(end_time * fs))]
                                preprocessor = Preprocessing(parameters=self.feature_parameters)
                                feature = preprocessor.feature_extraction(preprocessor=preprocessor, dataset=self, audio_raw=audio_raw)
                                features[point_idx] = np.reshape(feature, (-1,))
                    if save_features:
                        self.save_features_to_file(features, feature_file_addr)
                        # pickle.dump(features, open(feature_file_addr, 'wb'), 2)
            pickle.dump(data_list, open(datalist_pickle_file, 'wb'), 2)
        else:
            data_list = pickle.load(open(datalist_pickle_file, 'rb'))

        # normalization, val and test set using training mean and training std
        if self.normalization:
            mean_std_file_addr = os.path.join(self.feature_dir,
                                              'mean_std_time_res' + str(self.FLAGS.time_resolution) + '.json')
            if not tf.gfile.Exists(mean_std_file_addr):
                feature_buf = []
                batch_count = 0
                for training_point in tqdm(data_list['training'], desc='Computing training set mean and std'):
                    feature_idx = training_point.feature_idx
                    data_name = training_point.data_name
                    sub_dir = training_point.sub_dir
                    feature_file_addr = self.get_feature_file_addr(sub_dir, data_name)
                    features = self.read_features_to_nparray(feature_file_addr)

                    feature_buf.append(features[feature_idx])
                    batch_count += 1
                    if batch_count >= 512:
                        self.online_mean_variance(feature_buf)
                        feature_buf = []
                        batch_count = 0

                json.dump(obj=dict(
                    {'training_mean': self.training_mean.tolist(), 'training_std': self.training_std.tolist()}),
                          fp=open(mean_std_file_addr, 'wb'))
            else:
                training_statistics = json.load(open(mean_std_file_addr, 'r'))
                self.training_mean = np.reshape(training_statistics['training_mean'], (1, -1))
                self.training_std = np.reshape(training_statistics['training_std'], (1, -1))

        # count data point
        self.num_training_data = len(data_list['training'])
        self.num_validation_data = len(data_list['validation'])
        self.num_testing_data = len(data_list['testing'])
        return data_list


