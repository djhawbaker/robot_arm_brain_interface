#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
from BoardInterface import BoardInterface
#from EEGModels import EEGNet, DeepConvNet


def main():
    board_i = BoardInterface()

    board_i.start_stream()
    i = 0
    while True:
        try:
            # EEG data
            data = board_i.get_data(seconds=20)
            clean_data = board_i.denoise(data, "raw_data_" + str(i) + "_")
            board_i.write_data(data, "raw_data_" + str(i) + ".csv")
            # TODO get clean data to write
            # board_i.write_data(clean_data, "clean_data_" + str(i) + ".csv")
            board_i.process_input(data)

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
