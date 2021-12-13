#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

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
        """ Set the parameters for the board used to collect data

        :return: None
        """
        self.params.ip_port = 0
        self.params.serial_port = '/dev/serial/by-id/usb-Bluegiga_Low_Energy_Dongle_1-if00'
        self.params.mac_address = 'c0:b9:23:13:2c:be'
        self.params.other_info = ''
        self.params.serial_number = ''
        self.params.ip_address = ''
        self.params.ip_protocol = 0
        self.params.timeout = 0
        self.params.file = 'output.csv'

    def denoise(self, data, filename='denoise', save_after_plot=False, save_before_plot=False):
        """ Denoises the signal and saves a plot of it

        :param data: data from board to denoise
        :param filename: output base filename
        :param save_after_plot: Boolean to save the plot before processing or not
        :param save_before_plot: Boolean to save the plot after processing or not
        :return: processed data frame
        """
        df = pd.DataFrame(np.transpose(data))
        if save_before_plot:
            plt.figure(clear=True)
            df[self.eeg_channels].plot(subplots=True)
            plt.savefig(filename + '_before_processing.png')
            plt.close()

        for count, channel in enumerate(self.eeg_channels):
            DataFilter.perform_wavelet_denoising(data[channel], 'haar', 4)

        df = pd.DataFrame(np.transpose(data))
        if save_after_plot:
            plt.figure(clear=True)
            df[self.eeg_channels].plot(subplots=True)
            plt.savefig(filename + '_after_processing.png')
            plt.close()
        return df

    def start_stream(self):
        """ Start streaming data from the board

        :return: None
        """
        self.board.prepare_session()

        self.board.start_stream()  # use this for default options
        # board.start_stream(45000, args.streamer_self.params)
        print("################################################################################")
        print("###                               Streaming                                  ###")
        print("###                          Press Ctrl+c to stop                            ###")
        print("################################################################################")

    def end_stream(self):
        """ Stop streaming data from the board

        :return: None
        """
        self.board.stop_stream()
        self.board.release_session()
        print("\n")
        print("################################################################################")
        print("###                    Successfully Stopped Streaming                        ###")
        print("################################################################################")

    def get_data(self, samples=200):
        """ Get x number of samples from the board

        :param samples: The number of samples to collect at a time
        :return: The raw data collected
        """
        data = self.board.get_board_data(num_samples=samples)
        return data

    @staticmethod
    def read_file(filename):
        """ Reads a data signal file from disk

        :param filename: Name including path of file to open
        :return: Data
        """
        return DataFilter.read_file(file_name=filename)

    @staticmethod
    def write_data(data, file):
        """ Write data to a file

        :param data: The data to write
        :param file: Name including path of file to write
        :return: None
        """
        try:
            with open(file, 'w') as f:
                f.write(data)
        except Exception as exception:
            print("Caught: ", type(exception).__name__, ": ", exception.__str__(), ", while writing to ", file)
