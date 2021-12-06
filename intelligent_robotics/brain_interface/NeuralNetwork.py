#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
import matplotlib.pyplot as plt
import pandas as pd
import os

from ImportData import get_labels, get_data
from os import listdir
from os.path import isfile, isdir, join

import numpy as np
from numpy.random import shuffle
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics


class NeuralNetwork:
    def __init__(self, base_file_path):
        """ Initialize the Neural Network class

        :param base_file_path: Path to the root of the directory of the project
        """
        self.base_path = base_file_path
        self.X = []
        self.Y = []
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.dummy_y = None

        self.model = None
        self.df = None
        self.history = None

    @staticmethod
    def generate_filename(run, K, k):
        """ Generates the filename to save

        :param run: The run this data was linked with
        :param K: How many clusters
        :param k:
        :return: The filename
        """
        if not os.path.exists('output'):
            os.makedirs('output')

        filename = 'output/kmeans_' + str(K) + '_' + str(k) + '_' + str(run)
        return filename

    @staticmethod
    def get_unique_filename(base_path, base_name):
        """ Generate a unique filename. Increment the number appended to the end

        :param base_path: Path to where file will be saved
        :param base_name: Base of the filename to increment the number for
        :return: Filename
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        files = [f for f in listdir(base_path)]
        files.sort()
        nums = []
        for file in files:
            if base_name == file[:len(base_name)]:
                nums.append(int(file[len(base_name) + 1:]))

        try:
            num = max(nums) + 1
        except ValueError:
            num = 1

        return base_name + '_' + str(num)

    def create_model(self, save_summary=False):
        """ Create the neural network model to be trained

        :param save_summary: Boolean to save the model summary or not
        :return: The model
        """

        model = Sequential()
        model.add(Conv1D(8, 5, padding='same', input_shape=(400, 1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.4))

        # This layer was added to get more parameters.
        """
        model.add(Conv1D(48, 5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        """

        model.add(Conv1D(24, 5, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(4, activation='sigmoid'))

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model = model

        # Save summary to file
        if save_summary:
            self.save_summary('categorical_classifier', model)

        return model

    def load_model(self, path_to_model):
        """ Load the saved trained neural network model

        :param path_to_model: Location of the saved model
        :return: None
        """
        self.model = load_model(path_to_model)

    def load_data(self):
        """ Load the training data

        :return: None
        """
        single_path = self.base_path + '/data/single_blink/'
        double_path = self.base_path + '/data/double_blinks/'
        triple_path = self.base_path + '/data/three_blinks/'
        paths = [single_path, double_path, triple_path]
        labels = []

        for path in paths:
            # Get how many directories of data we have
            dirs = [d for d in listdir(path) if isdir(join(path, d))]
            dirs.sort()
            for y in dirs:
                data_path = path + y + '/output/'
                files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
                files.sort()
                for file in files:
                    data = get_data(data_path + file)
                    if data is not None:
                        self.X.append(data)
                new_labels = get_labels(path + y + '/labels.txt')
                labels += new_labels
                assert(len(files) == len(new_labels))

        self.Y = labels
        self.process_data()

    def load_test_data(self, path):
        """ Load the test training data
        NOTE Just for validating recorded test data

        :return: Data
        """
        return_data = []

        data_path = path + '/output/'
        files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        files.sort()
        for file in files:
            data = get_data(data_path + file)
            if data is not None:
                return_data.append(data)

        return return_data


    def process_labels(self):
        """ Format the one hot encoded training data labels

        :return: None
        """
        train_df = pd.DataFrame({'categories': self.train_y})
        test_df = pd.DataFrame({'categories': self.test_y})

        train_encoder = OneHotEncoder(handle_unknown='ignore')
        train_encoder_df = pd.DataFrame(train_encoder.fit_transform(train_df[['categories']]).toarray())
        final_train_df = train_df.join(train_encoder_df)
        final_train_df.drop('categories', axis=1, inplace=True)
        final_train_df.columns = ['Nothing', 'OneBlink', 'TwoBlinks', 'ThreeBlinks']
        print(final_train_df)

        test_encoder = OneHotEncoder(handle_unknown='ignore')
        test_encoder_df = pd.DataFrame(test_encoder.fit_transform(train_df[['categories']]).toarray())
        final_test_df = train_df.join(test_encoder_df)
        final_test_df.drop('categories', axis=1, inplace=True)
        final_test_df.columns = ['Nothing', 'OneBlink', 'TwoBlinks', 'ThreeBlinks']
        print(final_test_df)

    def process_data(self, test_percent=0.9):
        """ Process the training data for training
        Select equal amounts of each label type for training and testing
        Start by getting indices of each label

        :param test_percent: The percent of data that should be used for training
        :return: None
        """
        label_lists = [[i for i, value in enumerate(self.Y) if value == 0],
                       [i for i, value in enumerate(self.Y) if value == 1],
                       [i for i, value in enumerate(self.Y) if value == 2],
                       [i for i, value in enumerate(self.Y) if value == 3]]

        num_from_each = []
        for i in range(len(label_lists)):
            num_from_each.append(int(len(label_lists[i]) * test_percent))

        train_i = []
        test_i = []
        for i, num in enumerate(num_from_each):
            shuffle(label_lists[i])
            train_i += label_lists[i][:num]
            test_i += label_lists[i][num:]

        # Get values from indices
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in train_i:
            train_x.append(self.X[i])
            train_y.append(self.Y[i])

        for i in test_i:
            test_x.append(self.X[i])
            test_y.append(self.Y[i])

        arr = np.array(train_x)
        arr.shape = (len(arr), 400, 1)
        self.train_x = arr

        arr = np.array(test_x)
        arr.shape = (len(arr), 400, 1)
        self.test_x = arr

        # Setup one hot encoding
        self.train_y = np.array(train_y)
        self.test_y = np.array(test_y)

        self.process_labels()

    def train(self):
        """ Train the network

        :return: None
        """
        self.history = self.model.fit(self.train_x, self.train_y, epochs=400, shuffle=True)
        save_path = self.base_path + '/models/'
        unique_name = self.get_unique_filename(save_path, 'model')

        self.model.save(save_path + unique_name)

    def test(self):
        """ Test the network

        :return: None
        """
        eval_loss = self.model.evaluate(self.test_x, self.test_y)
        predictions = self.model.predict(self.test_x)
        y_preds = []
        for x in predictions:
            y_preds.append(np.unravel_index(np.argmax(x, axis=None), x.shape)[0])

        matrix = metrics.confusion_matrix(self.test_y, y_preds)
        self.save_summary(eval_loss, matrix)

    @staticmethod
    def prepare_data(data):
        """ Prepare the data of a single live input for classification

        :param data: The denoised data
        :return: NP Array of the data in the right shape
        """
        # Shift data to normalize across people
        mean = data[2].mean()[0]
        # Level that the network was trained on
        baseline = -923
        difference = int(abs(baseline) - abs(mean))
        if mean < baseline:
            data[2] += difference
        elif mean > baseline:
            data[2] -= difference

        # Get the right column
        alpha_channel = data[2].to_numpy()

        # Format it correctly
        alpha_channel.shape = (1, 400, 1)
        return alpha_channel

    @staticmethod
    def prepare_test_data(data):
        """ Prepare the data of a single recorded test input for classification

        :param data: The denoised data
        :return: NP Array of the data in the right shape
        """
        # Get the right column
        df = pd.DataFrame(np.transpose(data))
        mean = df.mean()[0]

        baseline = -923
        difference = int(abs(baseline) - abs(mean))
        if mean < baseline:
            df += difference
        elif mean > baseline:
            df -= difference

        alpha_channel = df.to_numpy()
        # Format it correctly
        alpha_channel.shape = (1, 400, 1)
        return alpha_channel

    def classify(self, data):
        """ Accept a single input and return the neural network classification

        :param data: Data to classify pandas data frame
        :return: classification: 0, 1, 2, 3
        """
        prediction = self.model.predict(data)
        print("prediction: ", prediction)
        y_preds = []
        for x in prediction:
            y_preds.append(np.unravel_index(np.argmax(x, axis=None), x.shape)[0])

        return y_preds[0]

    def save_summary(self, loss, confusion_matrix):
        """ Save the results of training the network

        :param loss: The network loss
        :param confusion_matrix: The confusion matrix
        :return: None
        """
        # TODO save this data
        print("History: ", self.history)
        print("Eval loss: ", loss)
        print(confusion_matrix)
