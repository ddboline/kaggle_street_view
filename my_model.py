#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl

import pandas as pd

from load_fn import load_data
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from sklearn.cross_validation import cross_val_score as k_fold_CV

def float32(k):
    return np.cast['float32'](k)

model = NeuralNet(
    layers=[ # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=200,  # we want to train this many epochs
    verbose=1,)

def train_model():
    imageSize = 400 # 20 x 20 pixels

    #Set location of data files , folders
    path = '.'

    labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(path))

    #Read training matrix
    xTrain = load_data("train", labelsInfoTrain, imageSize, path)

    #Read information about test data ( IDs ).
    labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(path))

    #Read test matrix
    xTest = load_data("test", labelsInfoTest, imageSize, path)

    yTrain = labelsInfoTrain['Class'].map(ord)

    xTrain, yTrain = map(float32, [xTrain, yTrain])

    cvAccuracy = np.mean(k_fold_CV(model, xTrain, yTrain, cv=2, scoring="accuracy"))

def get_submission():
    imageSize = 400 # 20 x 20 pixels

    #Set location of data files , folders
    path = '.'

    labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(path))

    #Read training matrix
    xTrain = load_data("train", labelsInfoTrain, imageSize, path)

    #Read information about test data ( IDs ).
    labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(path))

    #Read test matrix
    xTest = load_data("test", labelsInfoTest, imageSize, path)

    yTrain = labelsInfoTrain['Class'].map(ord)
    
    model.fit(xTrain, yTrain)
    yTest = model.predict(xTest)
    
    submit_df = labelsInfoTest.drop('Class')
    submit_df = submit_df.append({'Class': yTest})
    submit_df['Class'].map(chr)
    submit_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    train_model()
