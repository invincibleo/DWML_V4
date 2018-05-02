#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.08.17 11:02
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Dataset
# @Software: PyCharm Community Edition
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import json
import hashlib
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import scipy
import threading
import pandas as pd


class threadsafe_iter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class Dataset(object):

    def __init__(self, *args, **kwargs):
        self.data_list = kwargs.get('data_list', None)
        self.dataset_name = kwargs.get('dataset_name', "")
        self.dataset_dir = kwargs.get('dataset_dir', "")
        self.feature_dir = kwargs.get('feature_dir', "")
        self.dataset_list_dir = kwargs.get('dataset_list_dir', "")
        self.num_classes = int(kwargs.get('num_classes', 0))
        self.label_list = kwargs.get('label_list', [])
        self.FLAGS = kwargs.get('flag', '')
        self.preprocessing_methods = kwargs.get('preprocessing_methods', [])
        self.normalization = kwargs.get('normalization', True)
        self.training_mean = 0.0
        self.training_std = 1.0
        self.num_training_data = kwargs.get('num_training_data', 0)
        self.num_validation_data = kwargs.get('num_validation_data', 0)
        self.num_testing_data = kwargs.get('num_testing_data', 0)
        self.dimension = tuple([int(x) for x in kwargs.get('dimension', "").split(',')])

        self.feature_parameter_dir = kwargs.get('feature_parameter_dir', "")
        if self.feature_parameter_dir == "":
            self.feature_parameter_dir = self.FLAGS.parameter_dir
        self.feature_parameters = json.load(open(self.feature_parameter_dir + '/feature_parameters.json', 'r'))

        if self.dataset_list_dir == "":
            self.dataset_list_dir = os.path.join('tmp/dataset/', self.dataset_name)

        if self.feature_dir == "":
            self.feature_dir = os.path.join('tmp/feature/', self.dataset_name)



    def get_dataset_file_addr(self):
        file_name = "coding-" + self.FLAGS.coding + "_timeRes-" + str(self.FLAGS.time_resolution) + ".pickle"
        if not tf.gfile.Exists(self.dataset_list_dir):
            os.makedirs(self.dataset_list_dir)
        return os.path.join(self.dataset_list_dir, file_name)

    def get_feature_file_addr(self, sub_dir, data_name):
        hash_parameters = hashlib.sha1(str(self.feature_parameters)).hexdigest()
        feature_with_parameters_dir = os.path.join(self.feature_dir,
                                                   'timeRes-' + str(self.FLAGS.time_resolution) + '-' + hash_parameters)
        feature_parameter_file_name = 'feature_parameters.json'
        tf.gfile.MakeDirs(feature_with_parameters_dir)
        if not tf.gfile.Exists(os.path.join(feature_with_parameters_dir, feature_parameter_file_name)):
            tf.gfile.Copy(oldpath=os.path.join(self.feature_parameter_dir, feature_parameter_file_name),
                          newpath=os.path.join(feature_with_parameters_dir, feature_parameter_file_name),
                          overwrite=True)
        return os.path.join(feature_with_parameters_dir, sub_dir, data_name.split('.')[0] + '.csv')

    def save_features_to_file(self, features, feature_file_addr):
        features = pd.DataFrame.from_dict(features)
        features.to_csv(feature_file_addr, header=False, index=False)

    def read_features_to_nparray(self, feature_file_addr, entry_list=None):
        return np.transpose(pd.read_csv(feature_file_addr, header=None, index_col=None, parse_dates=True,
                                 usecols=entry_list).as_matrix())

    def online_mean_variance(self, new_batch_data):
        mean = np.zeros((1, self.dimension[-1])) #np.shape(new_batch_data)[2])
        M2 = np.zeros((1, self.dimension[-1]))

        for data_point in new_batch_data:
            xx = np.reshape(data_point, (-1, self.dimension[-1]))   #data_point
            for x in xx:
                self.num_training_data += 1
                delta = x - mean
                mean += delta / self.num_training_data
                delta2 = x - mean
                M2 += delta * delta2

        self.training_mean = mean
        if self.num_training_data < 2:
            self.training_std = float('nan')
        else:
            self.training_std = np.sqrt(M2 / self.num_training_data)

    def count_sets_data(self):
        # count data point
        self.num_training_data = len(self.data_list['training'])
        self.num_validation_data = len(self.data_list['validation'])
        self.num_testing_data = len(self.data_list['testing'])

    def dataset_normalization(self):
        # normalization, val and test set using training mean and training std
        mean_std_file_addr = os.path.join(self.feature_dir, 'mean_std_time_res' + str(self.FLAGS.time_resolution) + '.json')
        if not tf.gfile.Exists(mean_std_file_addr):
            if False: # predefine for future use to suit for big datasets
                all_features = dict()
                feature_buf = []
                batch_count = 0
                for training_point in tqdm(self.data_list['training'], desc='Computing training set mean and std'):
                    feature_idx = training_point.feature_idx
                    data_name = training_point.data_name
                    sub_dir = training_point.sub_dir
                    feature_file_addr = self.get_feature_file_addr(sub_dir, data_name)
                    if feature_file_addr not in all_features.keys() and len(all_features) < 4000:
                        features = pickle.load(open(feature_file_addr, 'rb'))
                        all_features[feature_file_addr] = features
                    elif feature_file_addr not in all_features.keys() and len(all_features) >= 4000:
                        rand_idx = np.random.choice(len(all_features))
                        pop_key = list(all_features.keys())[rand_idx]
                        all_features.pop(pop_key)

                    feature_buf.append(all_features[feature_file_addr][feature_idx])
                    batch_count += 1
                    if batch_count >= 128:
                        self.online_mean_variance(feature_buf)
                        feature_buf = []
                        batch_count = 0
            else:
                all_features = dict()
                feature_buf = []
                batch_count = 0
                for training_point in tqdm(self.data_list['training'], desc='Computing training set mean and std'):
                    feature_idx = training_point.feature_idx
                    data_name = training_point.data_name
                    sub_dir = training_point.sub_dir
                    feature_file_addr = self.get_feature_file_addr(sub_dir, data_name)

                    if feature_file_addr not in all_features.keys():
                        features = pickle.load(open(feature_file_addr, 'rb'))
                        all_features[feature_file_addr] = features

                    feature_buf.append(all_features[feature_file_addr][feature_idx])
                    batch_count += 1
                    if batch_count >= 128:
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

    def balance_data_list(self, data_list):
        class_count_buf_orign = np.zeros(data_list[0].label_content.shape)
        none_class_count = 0
        for data_point in data_list:
            label_content = data_point.label_content
            class_count_buf_orign += label_content
            if np.sum(label_content, axis=-1) == 0:
                none_class_count += 1

        max_class_num = np.max(class_count_buf_orign, axis=-1)
        min_class_num = np.min(class_count_buf_orign, axis=-1)
        num_to_add = max_class_num - class_count_buf_orign
        data_list_buf = data_list
        while(True):
            rand_idx = np.random.choice(len(data_list))
            data_point = data_list[rand_idx]
            label_content = data_point.label_content
            if not np.sum(label_content, axis=-1) == 0 and np.sum(np.sign(num_to_add[label_content == 1])) > 0:
                data_list_buf.append(data_point)
                num_to_add -= label_content

            if np.abs(np.std(num_to_add, axis=-1)) <= min_class_num:
                break

        none_class_count = 0
        class_count_buf = np.zeros(data_list[0].label_content.shape)
        for data_point in data_list_buf:
            label_content = data_point.label_content
            class_count_buf += label_content
            if np.sum(label_content, axis=-1) == 0:
                none_class_count += 1

        print("Class count: " + str(class_count_buf))
        return data_list_buf, class_count_buf_orign, none_class_count, class_count_buf

    @threadsafe_generator
    def generate_batch_data(self, category, batch_size=100, input_shape=(1, -1)):
        if category == 'training':
            working_list = self.data_list['training']
        elif category == 'validation':
            working_list = self.data_list['validation']
        elif category == 'testing':
            working_list = self.data_list['testing']

        num_data_files = len(working_list)
        print('\n' + category + ' generator initiated')
        generator_idx = 0
        while (1):
            random_perm = np.random.permutation(num_data_files)
            for i in range(0, int(np.floor(num_data_files/batch_size))):
                data_idx = random_perm[range(i*batch_size, (i+1)*batch_size)]
                data_points = [working_list[x] for x in data_idx]
                reference_dict = {}
                idx = 0
                for data_point in data_points:
                    data_name = data_point.data_name
                    sub_dir = data_point.sub_dir
                    feature_idx = data_point.feature_idx
                    label_content = data_point.label_content
                    if data_name not in reference_dict.keys():
                        reference_dict[data_name] = {'subdir': sub_dir, 'feature_entry': [feature_idx], 'label': [label_content], 'idx': [idx]}
                    else:
                        reference_dict[data_name]['feature_entry'].append(feature_idx)
                        reference_dict[data_name]['label'].append(label_content)
                        reference_dict[data_name]['idx'].append(idx)
                    idx = idx + 1

                features_for_batch = np.zeros(((batch_size, ) + input_shape))
                labels_for_batch = []
                for key, value in reference_dict.iteritems():
                    entry_list = value['feature_entry']
                    sub_dir = value['subdir']
                    idx_list = value['idx']
                    label_content_list = value['label']
                    data_name = key
                    feature_file_addr = self.get_feature_file_addr(sub_dir=sub_dir, data_name=data_name)
                    labels_for_batch = labels_for_batch + label_content_list
                    features = self.read_features_to_nparray(feature_file_addr, entry_list=entry_list)
                    idx = 0
                    for x in idx_list:
                        features_for_batch[x] = np.reshape(features[idx], input_shape)
                        idx = idx + 1

                labels_for_batch = np.array(labels_for_batch)
                labels_for_batch = np.reshape(labels_for_batch, (batch_size, -1))
                yield (features_for_batch, labels_for_batch)
                # print('\n' + category + ' generator yielded the batch %d' % generator_idx)
                generator_idx += 1

    @threadsafe_generator
    def generate_sequencial_batch_data(self, category, num_t_x, overlap=0.5, batch_size=100, input_shape=(1, -1)):
        if category == 'training':
            working_list = self.data_list['training']
        elif category == 'validation':
            working_list = self.data_list['validation']
        elif category == 'testing':
            working_list = self.data_list['testing']

        num_data_files = len(working_list)
        print('\n' + category + ' generator initiated')
        generator_idx = 0
        while (1):
            random_perm = np.random.permutation(int(num_data_files / (num_t_x * (1 - overlap))))
            for i in range(int(np.floor(num_data_files/batch_size))):
                window_start_point_idx = random_perm[range(i*batch_size, (i+1)*batch_size - 1)]
                batch_sequence_dict = {}
                for j in window_start_point_idx:
                    batch_sequence_dict[j] = {}
                    data_points = working_list[j:(j + num_t_x)]
                    idx = 0
                    for data_point in data_points:
                        data_name = data_point.data_name
                        sub_dir = data_point.sub_dir
                        feature_idx = data_point.feature_idx
                        label_content = data_point.label_content
                        if data_name not in batch_sequence_dict[j].keys():
                            batch_sequence_dict[j][data_name] = {'subdir': sub_dir,
                                                         'feature_entry': [feature_idx], 'label': [label_content],
                                                         'idx': [idx]}
                        else:
                            batch_sequence_dict[j][data_name]['feature_entry'].append(feature_idx)
                            batch_sequence_dict[j][data_name]['label'].append(label_content)
                            batch_sequence_dict[j][data_name]['idx'].append(idx)
                        idx = idx + 1

                features_for_batch = np.zeros((batch_size, num_t_x, ) + input_shape)
                labels_for_batch = []

                for wind_idx, wind_content in batch_sequence_dict.iteritems(): # get key=wind_idx value=wind_content
                    data_point_counter = 0
                    for key, value in wind_content.iteritems():
                        entry_list = value['feature_entry']
                        sub_dir = value['subdir']
                        idx_list = value['idx']
                        label_content_list = value['label']
                        data_name = key
                        feature_file_addr = self.get_feature_file_addr(sub_dir=sub_dir, data_name=data_name)
                        labels_for_batch.append(label_content_list)
                        features = self.read_features_to_nparray(feature_file_addr, entry_list=entry_list)
                        idx = 0
                        for x in idx_list:
                            features_for_batch[data_point_counter, x] = np.reshape(features[idx], input_shape)
                            idx = idx + 1
                        data_point_counter = data_point_counter + 1

                labels_for_batch = np.array(labels_for_batch)
                labels_for_batch = np.reshape(labels_for_batch, (batch_size, num_t_x, -1))
                labels_for_batch = list(np.swapaxes(labels_for_batch, 0, 1))
                yield (features_for_batch, labels_for_batch)
                # print('\n' + category + ' generator yielded the batch %d' % generator_idx)
                generator_idx += 1


    def get_batch_data(self, category, batch_size=100, input_shape=(1, -1)):
        X = []
        Y = []
        if category == 'training':
            working_list = self.data_list['training']
        elif category == 'validation':
            working_list = self.data_list['validation']
        elif category == 'testing':
            working_list = self.data_list['testing']

        num_data_files = len(working_list)
        data_point_list = []
        random_perm = np.random.permutation(num_data_files)
        while (1):
            for i in range(0, num_data_files):
                data_idx = random_perm[i]
                data_point = working_list[data_idx]
                data_point_list.append(data_point)
                data_name = data_point.data_name
                sub_dir = data_point.sub_dir
                label_content = data_point.label_content
                feature_idx = data_point.feature_idx

                feature_file_addr = self.get_feature_file_addr(sub_dir=sub_dir, data_name=data_name)

                features = pickle.load(open(feature_file_addr, 'rb'))

                feature = features[feature_idx]
                # if normalization then mean and std would not be 0 and 1 separately
                feature = np.reshape(feature, (-1, self.dimension[-1]))
                feature = (feature - self.training_mean) / self.training_std
                # feature = scipy.misc.imresize(feature, input_shape)
                feature = np.reshape(feature, (1, -1))
                feature = np.reshape(feature, input_shape)

                if not len(X) and not len(Y):
                    X = np.expand_dims(feature, axis=0)
                    Y = label_content #np.expand_dims(label_content, axis=0)
                else:
                    X = np.append(X, np.expand_dims(feature, axis=0), 0)
                    Y = np.append(Y, label_content, 0)

                if X.shape[0] >= batch_size:
                    return (X, Y, data_point_list)