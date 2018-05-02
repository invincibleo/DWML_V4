#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 14:51
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : util
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
import tensorflow as tf

def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
    
    Args:
    dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_data_file_path(dataset, data_name):
    data_list = dataset.data_list
    sub_dir = label_lists['subdir']
    full_path = os.path.join(base_dir, sub_dir, data_name)
    return full_path


def setup_keras():
    """Setup keras backend and parameters"""
    # Select Keras backend
    os.environ["KERAS_BACKEND"] = 'tensorflow'
    os.environ["HIP_VISIBLE_DEVICES"] = '0'
