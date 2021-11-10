#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""

import time
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *


class BoardInterface:
    def __init__(self):
        # Initialize the parameters
        self.params = BrainFlowInputParams()
        self.init_brainflow_params()

        self.board_id = 1  # GANGLION_BOARD
        self.master_board_id = self.board.get_board_id()
        self.board = BoardShim(self.board_id, self.params)
        self.sampling_rate = BoardShim.get_sampling_rate(self.master_board_id)

    def init_brainflow_params(self):
        BoardShim.enable_dev_board_logger()

        self.params.ip_port = 0
        self.params.serial_port = '/dev/ttyACM0'
        self.params.mac_address = 'c0:b9:23:13:2c:be'
        self.params.other_info = ''
        self.params.serial_number = ''
        self.params.ip_address = ''
        self.params.ip_protocol = 0
        self.params.timeout = 0
        self.params.file = 'output.csv'

    def denoise(self):
        pass

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

    def get_data(self):
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
        time.sleep(5)  # Get data once per second

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

    def write_data(self, data):
        DataFilter.write_file(data, self.params.file, 'a')
