#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""

from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    nn = NeuralNetwork()

    nn.create_model()
    nn.load_data()
    nn.train()
    #nn.test()
