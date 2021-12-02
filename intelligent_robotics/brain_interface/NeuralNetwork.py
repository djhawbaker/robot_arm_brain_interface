#!/usr/bin/env python3
"""
Class: Intelligent Robotics
Author: David Hawbaker
"""
import matplotlib.pyplot as plt
import copy
import pandas as pd
import os

from contextlib import redirect_stdout

from ImportData import get_labels, get_data
from os import listdir
from os.path import isfile, isdir, join

from tensorflow.keras.datasets.fashion_mnist import load_data
import skimage.measure
import numpy as np
from numpy import expand_dims, ones, zeros, vstack
from numpy.random import rand, randint, randn, seed, shuffle
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Flatten, Dropout, LeakyReLU, Reshape, Conv2DTranspose
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics


class NeuralNetwork:
    def __init__(self):
        self.X = []
        self.Y = []
        self.train_x = []
        self.train_y= []
        self.test_x = []
        self.test_y = []
        self.dummy_y = None

        self.model = None
        self.df = None

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

    def create_model(self, save_summary=False):
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

    def load_data(self):
        base = '/home/david/projects/psu_fall_2021/intelligent_robotics/brain_interface/data/'
        single_path = base + 'single_blink/'
        double_path = base + 'double_blinks/'
        triple_path = base + 'three_blinks/'
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
                # labels += get_labels(path + y + '/labels.txt')
                new_labels = get_labels(path + y + '/labels.txt')
                labels += new_labels
                assert(len(files) == len(new_labels))

        self.Y = labels
        # self.process_labels(labels)
        self.process_data()

    def process_labels(self):
        train_df = pd.DataFrame({'categories': self.train_y})
        test_df = pd.DataFrame({'categories': self.test_y})

        train_encoder = OneHotEncoder(handle_unknown='ignore')
        train_encoder_df = pd.DataFrame(train_encoder.fit_transform(train_df[['categories']]).toarray())
        final_train_df = train_df.join(train_encoder_df)
        final_train_df.drop('categories', axis=1, inplace=True)
        final_train_df.columns = ['Nothing', 'OneBlink', 'TwoBlinks', 'ThreeBlinks']
        print(final_train_df)
        #self.train_y = final_train_df

        test_encoder = OneHotEncoder(handle_unknown='ignore')
        test_encoder_df = pd.DataFrame(test_encoder.fit_transform(train_df[['categories']]).toarray())
        final_test_df = train_df.join(test_encoder_df)
        final_test_df.drop('categories', axis=1, inplace=True)
        final_test_df.columns = ['Nothing', 'OneBlink', 'TwoBlinks', 'ThreeBlinks']
        print(final_test_df)

    def process_data(self):
        # Select equal amounts of each label type for training and testing
        # Start by getting indices of each label
        test_percent = 0.9
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
        """
        encoder = LabelEncoder()
        encoder.fit(self.train_y)
        encoded_y = encoder.transform(self.train_y)
        self.train_y = to_categorical(encoded_y)

        encoder = LabelEncoder()
        encoder.fit(self.test_y)
        encoded_y = encoder.transform(self.test_y)
        self.test_y = to_categorical(encoded_y)
        """

    """
    @staticmethod
    def load_training_data():
        (train_x, _), (_, _) = load_data()
        data = expand_dims(train_x, axis=-1)
        data = data.astype('float32')
        data = data / 255.0
        return data

    def process_data(self, dataset):
        self.X = dataset[:,0:4].astype(float)
        Y = dataset[:,4]
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        self.dummy_y = to_categorical(encoded_Y)
    """

    def train(self):
        """
        estimator = KerasClassifier(build_fn=self.create_model, epochs=200, batch_size=5, verbose=0)
        kfold = KFold(n_splits=10, shuffle=True)
        results = cross_val_score(estimator, self.train_x, self.train_y, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        print('done')
        """

        history = self.model.fit(self.train_x, self.train_y, epochs=400, shuffle=True)
        #print("history: ", history)
        #estimator = KerasClassifier(build_fn=self.create_model, epochs=200, batch_size=5, verbose=0)
        #return estimator

    def test(self):
        eval_loss = self.model.evaluate(self.test_x, self.test_y)
        print("Eval loss: ", eval_loss)
        #kfold = KFold(n_splits=10, shuffle=True)
        #results = cross_val_score(estimator, self.X, self.dummy_y, cv=kfold)
        #print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        predictions = self.model.predict(self.test_x)
        #print("predictions: ", predictions)
        y_preds = []
        for x in predictions:
            y_preds.append(np.unravel_index(np.argmax(x, axis=None), x.shape)[0])

        #ind = np.unravel_index(np.argmax(predictions, axis=None), predictions.shape)
        #print("preds: ", ind)
        #y_pred = (predictions.argmax() > 0.5)
        #print("test_y: ", self.test_y)
        #print("y_pred: ", y_preds)
        matrix = metrics.confusion_matrix(self.test_y, y_preds) #predictions.argmax())
        print(matrix)

    def save_summary(self):
        pass

