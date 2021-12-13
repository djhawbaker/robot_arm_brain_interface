#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
from BoardInterface import BoardInterface
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    base_path = '/home/david/projects/psu_fall_2021/intelligent_robotics/brain_interface/'
    model_path = base_path + 'models/model_1'
    data_path = base_path + 'data/a/'

    nn = NeuralNetwork(base_path)
    nn.load_model(model_path)

    raw_data = nn.load_test_data(data_path)

    board_i = BoardInterface()
    i = 0
    for data in raw_data:
        #clean_data = board_i.denoise(data, "raw_data_" + str(i))
        # Uncomment to write data to file
        # board_i.write_data(data, "raw_data_" + str(i) + ".csv")
        # board_i.write_data(clean_data.to_string(), "clean_data_" + str(i) + ".csv")

        # Neural Network
        processed_data = nn.prepare_test_data(data)
        classification = nn.classify(processed_data)
        print("Classification ", str(i), ": ", classification)
        i += 0

