#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
import time
from brainflow.board_shim import BoardShim, LogLevels
from BoardInterface import BoardInterface
# from EEGModels import EEGNet, DeepConvNet


def main():
    board_i = BoardInterface()

    board_i.start_stream()
    i = 0
    while True:
        try:
            # EEG data
            seconds = 2
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
            time.sleep(seconds)  # Get data once per second

            data = board_i.get_data(samples=400)
            # TODO improve filename. ie read for existing files and increment
            clean_data = board_i.denoise(data, "raw_data_" + str(i))
            # board_i.write_data(data, "raw_data_" + str(i) + ".csv")
            board_i.write_data(clean_data.to_string(), "clean_data_" + str(i) + ".csv")
            # board_i.process_input(data)

            # Neural Network
            """
            nb_classes = 4
            model = DeepConvNet(nb_classes, Chans=4)
            model.compile()
            """
            i += 1

        except KeyboardInterrupt:
            board_i.end_stream()
            break


if __name__ == "__main__":
    main()
