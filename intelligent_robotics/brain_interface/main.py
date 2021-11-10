#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
from BoardInterface import BoardInterface
from EEGModels import EEGNet


def main():

    board_i = BoardInterface()

    board_i.start_stream()

    while True:
        try:
            data = board_i.get_data()
            clean_data = board_i.denoise(data)
            board_i.write_data(data, "raw_data.csv")
            board_i.write_data(clean_data, "clean_data.csv")

        except KeyboardInterrupt:
            board_i.end_stream()
            break


if __name__ == "__main__":
    main()
