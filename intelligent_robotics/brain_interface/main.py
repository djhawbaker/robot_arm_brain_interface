#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
import time
from brainflow.board_shim import BoardShim, LogLevels
from BoardInterface import BoardInterface
from NeuralNetwork import NeuralNetwork
from ServoInterface import ServoInterface


def main():
    nn = NeuralNetwork()
    model_path = '/home/david/projects/psu_fall_2021/intelligent_robotics/brain_interface/models/model_1'
    nn.load_model(model_path)

    si = ServoInterface()

    board_i = BoardInterface()

    board_i.start_stream()
    i = 0
    while True:
        try:
            # EEG data
            seconds = 2
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
            time.sleep(seconds)

            data = board_i.get_data(samples=400)
            # TODO improve filename. ie read for existing files and increment
            clean_data = board_i.denoise(data, "raw_data_" + str(i))
            # board_i.write_data(data, "raw_data_" + str(i) + ".csv")
            # board_i.write_data(clean_data.to_string(), "clean_data_" + str(i) + ".csv")
            # board_i.process_input(data)

            # Neural Network
            processed_data = nn.prepare_data(clean_data)
            classification = nn.classify(processed_data)
            print("Classification: ", classification)

            si.interpret_classification(classification)
            i += 1

        except KeyboardInterrupt:
            board_i.end_stream()
            break


if __name__ == "__main__":
    main()
