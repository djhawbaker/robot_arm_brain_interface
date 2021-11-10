#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""

import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *

matplotlib.use('Agg')

class BoardInterface:
    def __init__(self):
        BoardShim.enable_dev_board_logger()
        # Initialize the parameters
        self.params = BrainFlowInputParams()
        self.init_brainflow_params()

        self.board_id = BoardIds.GANGLION_BOARD.value
        self.board = BoardShim(self.board_id, self.params)

        self.master_board_id = self.board.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(self.master_board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)

    def init_brainflow_params(self):

        self.params.ip_port = 0
        self.params.serial_port = '/dev/serial/by-id/usb-Bluegiga_Low_Energy_Dongle_1-if00'
        self.params.mac_address = 'c0:b9:23:13:2c:be'
        self.params.other_info = ''
        self.params.serial_number = ''
        self.params.ip_address = ''
        self.params.ip_protocol = 0
        self.params.timeout = 0
        self.params.file = 'output.csv'

    def denoise(self, data, filename):
        df = pd.DataFrame(np.transpose(data))
        plt.figure()
        df[self.eeg_channels].plot(subplots=True)
        plt.savefig(filename + 'before_processing.png')
        plt.close()

        # demo for de-noising, apply different methods to different channels for demo
        for count, channel in enumerate(self.eeg_channels):
            # first of all you can try simple moving median or moving average with different window size
            if count == 0:
                DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEAN.value)
            elif count == 1:
                DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEDIAN.value)
            # if methods above dont work for your signal you can try wavelet based denoising
            # feel free to try different functions and decomposition levels
            elif count == 2:
                DataFilter.perform_wavelet_denoising(data[channel], 'db6', 3)
            elif count == 3:
                DataFilter.perform_wavelet_denoising(data[channel], 'bior3.9', 3)
            elif count == 4:
                DataFilter.perform_wavelet_denoising(data[channel], 'sym7', 3)
            elif count == 5:
                # with synthetic board this one looks like the best option, but it depends on many circumstances
                DataFilter.perform_wavelet_denoising(data[channel], 'coif3', 3)

        df = pd.DataFrame(np.transpose(data))
        plt.figure()
        df[self.eeg_channels].plot(subplots=True)
        plt.savefig(filename + 'after_processing.png')
        plt.close()

    def start_stream(self):
        self.board.prepare_session()

        self.board.start_stream()  # use this for default options
        # board.start_stream(45000, args.streamer_self.params)
        print("################################################################################")
        print("###                               Streaming                                  ###")
        print("###                          Press Ctrl+c to stop                            ###")
        print("################################################################################")

    def end_stream(self):
        self.board.stop_stream()
        self.board.release_session()
        print("\n")
        print("################################################################################")
        print("###                    Successfully Stopped Streaming                        ###")
        print("################################################################################")

    def get_data(self, seconds=5):
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
        time.sleep(seconds)  # Get data once per second

        data = self.board.get_board_data(num_samples=200)
        # get latest 256 packages or less, doesnt remove them from internal buffer
        # data = board.get_current_board_data (256)

        # get 200 data samples and remove it from internal buffer
        return data

    def process_input(self, data):
        eeg_channels = BoardShim.get_eeg_channels(int(self.master_board_id))
        bands = DataFilter.get_avg_band_powers(data, eeg_channels, self.sampling_rate, True)
        feature_vector = np.concatenate((bands[0], bands[1]))
        print(feature_vector)

        # calc concentration
        concentration_params = BrainFlowModelParams(BrainFlowMetrics.CONCENTRATION.value,
                                                    BrainFlowClassifiers.KNN.value)
        concentration = MLModel(concentration_params)
        concentration.prepare()
        concentrate_result = concentration.predict(feature_vector)
        print('Concentration: %f' % concentrate_result)
        concentration.release()

        # calc relaxation
        relaxation_params = BrainFlowModelParams(BrainFlowMetrics.RELAXATION.value,
                                                 BrainFlowClassifiers.REGRESSION.value)
        relaxation = MLModel(relaxation_params)
        relaxation.prepare()
        relax_result = relaxation.predict(feature_vector)
        print('Relaxation: %f' % relax_result)
        relaxation.release()

        return concentrate_result, relax_result

    def write_data(self, data, file):
        # TODO use proper except handling
        try:
            DataFilter.write_file(data, file, 'a')
        except:
            print("Empty buffer error while writing data to file: ", file)
