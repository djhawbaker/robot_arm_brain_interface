#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker

This is the main program. Run it on the Raspberry Pi once the EEG and robot arm are all connected
It will read the brain waves of the user and interpret 3 blink commands to move the fingers of the
robot hand.
"""
import time
from brainflow.board_shim import BoardShim, LogLevels
from BoardInterface import BoardInterface
from NeuralNetwork import NeuralNetwork
from ServoInterface import ServoInterface


def main():
    # Modify this to set the root directory and model to use
    base_path = '/your/full/path/to/directory'
    # Select which trained model to use
    model_path = base_path + 'models/model_2'

    nn = NeuralNetwork(base_path)
    nn.load_model(model_path)

    si = ServoInterface()

    board_i = BoardInterface()

    board_i.start_stream()
    i = 0
    while True:
        try:
            # EEG data
            seconds = 2
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Give your blink command now')
            time.sleep(seconds)

            data = board_i.get_data(samples=400)
            # TODO improve filename. ie read for existing files and increment
            clean_data = board_i.denoise(data, "raw_data_" + str(i))
            # Uncomment to write csv data to file
            # board_i.write_data(data, "raw_data_" + str(i) + ".csv")
            # board_i.write_data(clean_data.to_string(), "clean_data_" + str(i) + ".csv")

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
