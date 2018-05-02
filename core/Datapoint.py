#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 24.08.17 16:49
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Datapoint
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Datapoint(object):
    def __init__(self, *args, **kwargs):
         self.data_name = kwargs.get('data_name', "")
         self.sub_dir = kwargs.get('sub_dir', "")
         self.label_name = kwargs.get('label_name', "")
         self.data_content = kwargs.get('data_content', None)
         self.label_content = kwargs.get('label_content', None)