#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25.08.17 10:19
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : TimeserisePoint
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.Datapoint import Datapoint


class TimeseriesPoint(Datapoint):
    
    def __init__(self, *args, **kwargs):
        super(TimeseriesPoint, self).__init__(self, *args, **kwargs)
        try:
            self.start_time = float(kwargs.get('start_time'))
            self.end_time = float(kwargs.get('end_time'))
        except TypeError:
            print('Please input start and end time!')

        self.duration = self.end_time - self.start_time
        self.feature_idx = kwargs.get('feature_idx', None)


class AudioPoint(TimeseriesPoint):
    
    def __init__(self, *args, **kwargs):
        super(AudioPoint, self).__init__(self, *args, **kwargs)
        self.fs = int(kwargs.get('fs', 0))
        self.extension = kwargs.get('extension', "")


class PicturePoint(Datapoint):
    def __init__(self, *args, **kwargs):
        super(PicturePoint, self).__init__(self, *args, **kwargs)
        self.extension = kwargs.get('extension', "")
        self.pic_shape = kwargs.get('pic_shape', [])