#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""

from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork('/home/david/projects/psu_fall_2021/intelligent_robotics/brain_interface/')

    nn.create_model()
    nn.load_data()
    nn.train()
    nn.test()
