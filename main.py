#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 07.08.17 16:15
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : main
# @Software: PyCharm Community Edition

import argparse
import sys
import json
import datetime

from application.Dataset_AVEC2016_V2 import *
from application.LearnerLSTMReg_V2 import *
from application.Evaluator_AVEC2016 import *


def main(_):
    dataset = Dataset_AVEC2016_V2(dataset_dir=FLAGS.data_dir,
                                  normalization=False,
                                  using_existing_features=False,
                                  flag=FLAGS)

    learner = LearnerLSTMReg(dataset=dataset,
                             learner_name='LSTMReg',
                             flag=FLAGS)

    evaluator = Evaluator_AVEC2016()

    learner.learn()
    truth, prediction = learner.predict()
    evaluator.evaluate(truth, prediction)
    results = evaluator.results()

    print(results)
    results_dir_addr = 'tmp/results/'
    current_time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not tf.gfile.Exists(results_dir_addr):
        tf.gfile.MakeDirs(results_dir_addr)
    hash_FLAGS = hashlib.sha1(str(FLAGS)).hexdigest()
    results_file_dir = os.path.join(results_dir_addr, dataset.dataset_name, hash_FLAGS)
    if not tf.gfile.Exists(results_file_dir):
        tf.gfile.MakeDirs(results_file_dir)
        json.dump(results, open(results_file_dir + '/results_' + current_time_str + '.json', 'wb'), indent=4)
        json.dump(zip(truth[:, 0].tolist(), prediction[:, 0].tolist()),
                  open(results_file_dir + '/results_' + current_time_str + '_0.json', 'a'), indent=4)
        json.dump(zip(truth[:, 1].tolist(), prediction[:, 1].tolist()),
                  open(results_file_dir + '/results_' + current_time_str + '_1.json', 'a'), indent=4)
        with open(results_file_dir + 'FLAGS_' + current_time_str + '.txt', 'wb') as f:
            f.write(str(FLAGS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./AVEC2016',
        help='Path to folders of labeled audios.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=1000,
        help="""\
        How many images to test on. This test set is only used once, to evaluate
        the final accuracy of the model after training completes.
        A value of -1 causes the entire test set to be used, which leads to more
        stable results across runs.\
        """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
        How many images to use in an evaluation batch. This validation set is
        used much more often than the test set, and is an early indicator of how
        accurate the model is during training.
        A value of -1 causes the entire validation set to be used, which leads to
        more stable results across training iterations, but may be slower on large
        training sets.\
        """
    )
    parser.add_argument(
        '--time_resolution',
        type=float,
        default=0.04,
        help="""\
        The hop of the FFT in sec.\
        """
    )
    parser.add_argument(
        '--fs',
        type=int,
        default=44100,
        help="""\
        The sampling frequency if an time-series signal is given\
        """
    )
    parser.add_argument(
        '--drop_out_rate',
        type=float,
        default=0.4,
        help="""\
        \
        """
    )
    parser.add_argument(
        '--coding',
        type=str,
        default='number',
        help="""\
        one hot encoding: onehot, k hot encoding: khot, continues value: number
        \
        """
    )
    parser.add_argument(
        '--parameter_dir',
        type=str,
        default="parameters",
        help="""\
        parameter folder
        \
        """
    )
    parser.add_argument(
        '--dimension',
        type=str,
        default="25, 1024",
        help="""\
        input dimension to the model
        \
        """
    )
    parser.add_argument(
        '--output_activation',
        type=str,
        default="linear",
        help="""\
        output activation function, e.g. sigmoid, tanh, linear
        \
        """
    )
    parser.add_argument(
        '--regularization_constant',
        type=float,
        default=0.0001,
        help="""\
        l2 regularizer weight
        \
        """
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="""\
        number of epochs
        \
        """
    )
    parser.add_argument(
        '--learning_rate_decay',
        type=bool,
        default=True,
        help="""\
        if learning rate decay
        \
        """
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help="""\
        how long the annotation is delay to the features
        \
        """
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)