#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
import pandas as pd


def get_labels(filename):
    """ Get the labels for the directory

    :param filename: Name of file including path
    :return: list of labels
    """
    f = open(filename, 'r')
    line = f.readline()
    labels = line.split(' ')
    f.close()
    return labels


def get_data(filename):
    """ Get the channel data from the csv file

    :param filename: Name of file including path
    :return: column data
    """
    df = pd.read_csv(filename)
    column_data = df['2']  # you can also use df['column_name']
    return column_data
