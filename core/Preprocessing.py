#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28.08.17 13:46
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Preprocessing
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import librosa
import scipy
import numpy
import math
from core.GeneralFileAccessor import GeneralFileAccessor


class Preprocessing(object):
    def __init__(self, *args, **kwargs):
        self.eps = numpy.spacing(1)
        self.parameters = {
            'general': {
                'fs': 44100,
                'win_length_samples': int(0.04 * 44100),
                'hop_length_samples': int(0.02 * 44100),
            },
            'mfcc': {
                'mono': True,  # [True, False]
                'window': 'hamming_asymmetric',  # [hann_asymmetric, hamming_asymmetric]
                'spectrogram_type': 'power',  # [magnitude, power]
                'n_mfcc': 20,  # Number of MFCC coefficients
                'n_mels': 40,  # Number of MEL bands used
                'normalize_mel_bands': False,  # [True, False]
                'n_fft': 2048,  # FFT length
                'fmin': 0,  # Minimum frequency when constructing MEL bands
                'fmax': 22050,  # Maximum frequency when constructing MEL band
                'htk': True,  # Switch for HTK-styled MEL-frequency equation
                'log': True,  # Logarithmic
            },
            'mfcc_delta': {
                'width': 9,
                'dependency_method': 'mfcc',
            },
            'mfcc_acceleration': {
                'width': 9,
                'dependency_method': 'mfcc',
            },
            'mel': {
                'mono': True,  # [True, False]
                'window': 'hamming_asymmetric',  # [hann_asymmetric, hamming_asymmetric]
                'spectrogram_type': 'power',  # [magnitude, power]
                'n_mels': 40,  # Number of MEL bands used
                'normalize_mel_bands': False,  # [True, False]
                'n_fft': 2048,  # FFT length
                'fmin': 0,  # Minimum frequency when constructing MEL bands
                'fmax': 22050,  # Maximum frequency when constructing MEL band
                'htk': True,  # Switch for HTK-styled MEL-frequency equation
                'log': True,  # Logarithmic
            }
        }
        self.parameters.update(kwargs.get('parameters', {}))

    def _window_function(self, N, window_type='hamming_asymmetric'):
        """Window function

        Parameters
        ----------
        N : int
            window length

        window_type : str
            window type
            (Default value='hamming_asymmetric')
        Raises
        ------
        ValueError:
            Unknown window type

        Returns
        -------
            window function : array
        """

        # Windowing function
        if window_type == 'hamming_asymmetric':
            return scipy.signal.hamming(N, sym=False)
        elif window_type == 'hamming_symmetric':
            return scipy.signal.hamming(N, sym=True)
        elif window_type == 'hann_asymmetric':
            return scipy.signal.hann(N, sym=False)
        elif window_type == 'hann_symmetric':
            return scipy.signal.hann(N, sym=True)
        else:
            message = '{name}: Unknown window type [{window_type}]'.format(
                name=self.__class__.__name__,
                window_type=window_type
            )
            raise ValueError(message)

    def _spectrogram(self, y,
                     n_fft=1024,
                     win_length_samples=0.04,
                     hop_length_samples=0.02,
                     window=scipy.signal.hamming(1024, sym=False),
                     center=True,
                     spectrogram_type='magnitude'):
        """Spectrogram

        Parameters
        ----------
        y : numpy.ndarray
            Audio data
        n_fft : int
            FFT size
            Default value "1024"
        win_length_samples : float
            Window length in seconds
            Default value "0.04"
        hop_length_samples : float
            Hop length in seconds
            Default value "0.02"
        window : array
            Window function
            Default value "scipy.signal.hamming(1024, sym=False)"
        center : bool
            If true, input signal is padded so to the frame is centered at hop length
            Default value "True"
        spectrogram_type : str
            Type of spectrogram "magnitude" or "power"
            Default value "magnitude"

        Returns
        -------
        np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
            STFT matrix

        """

        if spectrogram_type == 'magnitude':
            return numpy.abs(librosa.stft(y + self.eps,
                                          n_fft=n_fft,
                                          win_length=win_length_samples,
                                          hop_length=hop_length_samples,
                                          center=center,
                                          window=window))
        elif spectrogram_type == 'power':
            return numpy.abs(librosa.stft(y + self.eps,
                                          n_fft=n_fft,
                                          win_length=win_length_samples,
                                          hop_length=hop_length_samples,
                                          center=center,
                                          window=window)) ** 2
        else:
            message = '{name}: Unknown spectrum type [{spectrogram_type}]'.format(
                name=self.__class__.__name__,
                spectrogram_type=spectrogram_type
            )
            raise ValueError(message)

    def mel(self, audio_raw):
        """Mel-band energies

        Parameters
        ----------
        audio_raw : numpy.ndarray
            Audio data

        Returns
        -------
        list of numpy.ndarrays
            List of feature matrices, feature matrix per audio channel
        """

        # window func
        window = self._window_function(N=self.parameters['general'].get('win_length_samples'),
                                       window_type=self.parameters['mel'].get('window'))

        # mel filter banks
        mel_basis = librosa.filters.mel(sr=self.parameters['general'].get('fs'),
                                        n_fft=self.parameters['mel'].get('n_fft'),
                                        n_mels=self.parameters['mel'].get('n_mels'),
                                        fmin=self.parameters['mel'].get('fmin'),
                                        fmax=self.parameters['mel'].get('fmax'),
                                        htk=self.parameters['mel'].get('htk'))

        if self.parameters['mel'].get('normalize_mel_bands'):
            mel_basis /= numpy.max(mel_basis, axis=-1)[:, None]

        # mel-freq spectrum
        # feature_matrix = []
        # for channel in range(0, audio_raw.shape[0]):
        channel = 0
        spectrogram_ = self._spectrogram(
            y=audio_raw[channel, :],
            n_fft=self.parameters['mel'].get('n_fft'),
            win_length_samples=self.parameters['general'].get('win_length_samples'),
            hop_length_samples=self.parameters['general'].get('hop_length_samples'),
            spectrogram_type=self.parameters['mel'].get('spectrogram_type') if 'spectrogram_type' in self.parameters['mel'] else 'power',
            center=True,
            window=window
        )

        mel_spectrum = numpy.dot(mel_basis, spectrogram_)
        if self.parameters['mel'].get('log'):
            mel_spectrum = numpy.log(mel_spectrum + self.eps)
            mel_spectrum = mel_spectrum.T
            # feature_matrix.append(mel_spectrum)

        return mel_spectrum

    def mfcc(self, audio_raw, plot=False):
        """Static MFCC

        Parameters
        ----------
        audio_raw : numpy.ndarray
            Audio data

        Returns
        -------
        list of numpy.ndarrays
            List of feature matrices, feature matrix per audio channel

        """

        window = self._window_function(N=self.parameters['general'].get('win_length_samples'),
                                       window_type=self.parameters['mfcc'].get('window'))

        mel_basis = librosa.filters.mel(sr=self.parameters['general'].get('fs'),
                                        n_fft=self.parameters['mfcc'].get('n_fft'),
                                        n_mels=self.parameters['mfcc'].get('n_mels'),
                                        fmin=self.parameters['mfcc'].get('fmin'),
                                        fmax=self.parameters['mfcc'].get('fmax'),
                                        htk=self.parameters['mfcc'].get('htk'))

        if self.parameters['mfcc'].get('normalize_mel_bands'):
            mel_basis /= numpy.max(mel_basis, axis=-1)[:, None]

        # feature_matrix = []
        # for channel in range(0, audio_raw.shape[0]):
        channel = 0
        # Calculate Static Coefficients
        spectrogram_ = self._spectrogram(
            y=audio_raw[channel, :],
            n_fft=self.parameters['mfcc'].get('n_fft'),
            win_length_samples=self.parameters['general'].get('win_length_samples'),
            hop_length_samples=self.parameters['general'].get('hop_length_samples'),
            spectrogram_type=self.parameters['mfcc'].get('spectrogram_type') if 'spectrogram_type' in self.parameters['mfcc'] else 'power',
            center=True,
            window=window
        )

        mel_spectrum = numpy.dot(mel_basis, spectrogram_)  # shape=(d, t)

        mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum),
                                    n_mfcc=self.parameters['mfcc'].get('n_mfcc'))
        mfcc = mfcc.T
            # feature_matrix.append(mfcc.T)

        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(1,2,1)
            plt.imshow(numpy.reshape(librosa.logamplitude(mel_spectrum), (self.parameters['mfcc'].get('n_mels'), -1)))
            plt.subplot(1,2,2)
            plt.imshow(numpy.reshape(mfcc, (self.parameters['mfcc'].get('n_mfcc'), -1)))
            plt.show()


        return mfcc

    def normalization(self, data):
        """Normalize feature matrix with internal statistics of the class

        Parameters
        ----------
        feature_container : numpy.ndarray [shape=(frames, number of feature values)]
            Feature matrix to be normalized
        channel : int
            Feature channel
            Default value "0"

        Returns
        -------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Normalized feature matrix

        """
        data_mean = numpy.mean(data, axis=1, keepdims=True)
        data_std = numpy.std(data, axis=1, keepdims=True)
        return (data - data_mean) / data_std


    @staticmethod
    def audio_event_roll(meta_file=None, meta_content=None, label_list=[], time_resolution=0.04):
        if meta_content is None:
            meta_file_content = GeneralFileAccessor(file_path=meta_file).read()
        elif meta_file is None:
            meta_file_content = list(meta_content)

        end_times = numpy.array([float(x[-2]) for x in meta_file_content])
        max_offset_value = numpy.max(end_times, 0)

        event_roll = numpy.zeros((int(math.floor(max_offset_value / time_resolution)), len(label_list)))
        start_times = []
        end_times = []

        for line in meta_file_content:
            label_name = line[-1]
            label_idx = label_list.index(label_name)
            event_start = float(line[-3])
            event_end = float(line[-2])

            onset = int(math.floor(event_start / time_resolution))
            offset = int(math.floor(event_end / time_resolution))

            event_roll[onset:offset, label_idx] = 1
            start_times.append(event_start)
            end_times.append(event_end)

        return event_roll

    @staticmethod
    def feature_extraction(preprocessor ,dataset, audio_raw):
        # feature extraction
        audio_raw = numpy.reshape(audio_raw, (1, -1))
        preprocessing = preprocessor
        preprocessing_func = audio_raw
        for preprocessing_method in dataset.preprocessing_methods:
            preprocessing_func = eval('preprocessing.' + preprocessing_method)(preprocessing_func)

        return preprocessing_func

