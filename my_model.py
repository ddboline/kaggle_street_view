#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl

import pandas as pd

from load_fn import load_data

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import cross_val_score as k_fold_CV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

ORD_VALUES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

def transform_str_to_feature(st):
    ordval = ord(st)
    ordidx = ORD_VALUES.index(ordval)
    n = len(ORD_VALUES)//2
    return (ordidx - n)/float(n)

def transform_feature_to_str(ft):
    n = len(ORD_VALUES)//2
    idx = int(ft*n + n)
    ordval = ORD_VALUES[idx]
    return chr(ordval)

def float32(k):
    return np.cast['float32'](k)


def train_nn_model():
    from lasagne import layers
    from lasagne.updates import nesterov_momentum
    from nolearn.lasagne import NeuralNet

    model = NeuralNet(
        layers=[ # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),],
        # layer parameters:
        input_shape=(None, 400),  # 96x96 input pixels per batch
        hidden_num_units=50,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=1,  # 30 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=200,  # we want to train this many epochs
        verbose=1,)

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

    yTrain = labelsInfoTrain['Class'].map(transform_str_to_feature)

    print yTrain

    xtrain, xtest, ytrain, ytest = train_test_split(xTrain, yTrain, test_size=0.5)

    xtrain, xtest, ytrain, ytest = map(float32, [xtrain, xtest, ytrain, ytest])

    print xtrain.shape, xtest.shape, ytrain.shape, ytest.shape
    model.fit(xtrain, ytrain)
    ytest_pred = model.predict(xtest)
    print model.accuracy_score(ytest_pred,ytest)

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

    xtrain, xtest, ytrain, ytest = train_test_split(xTrain, yTrain, test_size=0.5)

    model = RandomForestClassifier(n_estimators=400)
    model.fit(xtrain, ytrain)
    print model.score(xtest, ytest)
    ytest_pred = model.predict(xtest)
    print accuracy_score(ytest_pred,ytest)

def test_knn_model():
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

    xtrain, xtest, ytrain, ytest = train_test_split(xTrain, yTrain, test_size=0.5)
    start = time.time()
    tuned_parameters = [{"n_neighbors":list(range(1,5))}]
    clf = GridSearchCV( model, tuned_parameters, cv=5, scoring="accuracy")
    clf.fit(xtrain, ytrain)
    print clf.grid_scores_
    print time.time() - start, "seconds elapsed"

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
    
    model = RandomForestClassifier(n_estimators=200)
    model.fit(xTrain, yTrain)
    yTest = model.predict(xTest)
    
    submit_df = labelsInfoTest.drop('Class')
    submit_df = submit_df.append({'Class': yTest})
    submit_df['Class'].map(chr)
    submit_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    train_model()
