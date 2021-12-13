#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker

Use this file to read training data and labels from disk
"""


def get_labels(filename):
    """ Get the labels for the directory

    :param filename: Name of file including path
    :return: list of labels
    """
    with open(filename, 'r') as f:
        line = f.readline()
        line = line.replace('\n', '')
        labels = line.split(' ')

    labels = list(map(int, labels))

    return labels


def get_data(filename):
    """ Get the channel data from the csv file

    :param filename: Name of file including path
    :return: column data
    """
    column_data = []
    with open(filename, 'r') as f:
        column = 3
        first_row = True
        lines = f.readlines()
        for line in lines:
            if not first_row:
                row = line.split(' ')
                column_data.append(int(float(row[column])))
            first_row = False

    return column_data
