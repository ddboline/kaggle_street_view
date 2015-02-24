#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl

import pandas as pd

from load_fn import load_data

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import cross_val_score as k_fold_CV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

ORD_VALUES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
NORD = len(ORD_VALUES)

def transform_str_to_feature(st):
    #return ord(st)
    ordval = ord(st)
    ordidx = ORD_VALUES.index(ordval)
    return (ordidx-NORD//2)/float(NORD//2)

def transform_feature_to_str(ft):
    #return chr(ft)
    idx = NORD//2 + int(ft * NORD//2)
    if idx < 0 :
        idx = 0
    if idx >= NORD:
        idx = ORD_VALUES-1
    ordval = ORD_VALUES[idx]
    return chr(ordval)

def transform_from_classes(inp):
    y = np.zeros((inp.shape[0], NORD), dtype=np.float32)
    for (index, Class) in enumerate(inp):
        cidx = ORD_VALUES.index(ord(Class))
        y[index, cidx] = 1.0
    return y

def transform_to_class(yinp):
    return np.array(map(lambda x: chr(ORD_VALUES[x]), np.argmax(yinp, axis=1)))

def float32(k):
    return np.cast['float32'](k)

def load_train_test_data():
    imageSize = 400 # 20 x 20 pixels

    #Set location of data files , folders
    path = '.'

    labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(path))

    #Read training matrix
    xTrain = load_data("train", labelsInfoTrain, imageSize, path)

    yTrain = transform_from_classes(labelsInfoTrain['Class'])

    print xTrain.shape, yTrain.shape
    print xTrain.dtype, yTrain.dtype

    #Read information about test data ( IDs ).
    labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(path))

    #Read test matrix
    xTest = load_data("test", labelsInfoTest, imageSize, path)

    return xTrain, yTrain, xTest, labelsInfoTest

def train_nn_model():
    imageSize = 400 # 20 x 20 pixels

    from lasagne import layers
    from lasagne.updates import nesterov_momentum, sgd
    from nolearn.lasagne import NeuralNet

    model = NeuralNet(layers=[('input', layers.InputLayer),
                              ('hidden', layers.DenseLayer),
                              ('output', layers.DenseLayer),],
        # layer parameters:
        input_shape=(None, 400),  # 96x96 input pixels per batch
        hidden_num_units=100,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=62,  # 30 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=400,  # we want to train this many epochs
        verbose=1,)

    xTrain, yTrain, xTest, labelsInfoTest = load_train_test_data()

    print xTrain.shape, yTrain.shape
    print xTrain.dtype, yTrain.dtype

    model.fit(xTrain, yTrain)
    #ytest_pred = model.predict(xTrain)
    print model.score(xTrain, yTrain)
    #print accuracy_score(ytest_pred, yTrain)
    return model

def train_model():
    xTrain, yTrain, Xtest, labelsInfoTest = load_train_test_data()

    xtrain, xtest, ytrain, ytest = train_test_split(xTrain, yTrain, test_size=0.5)

    model = RandomForestClassifier(n_estimators=800, n_jobs=-1)
    model.fit(xtrain, ytrain)
    print model.score(xtest, ytest)
    ytest_pred = model.predict(xtest)
    print accuracy_score(ytest_pred, ytest)

def train_knn_model():
    xTrain, yTrain, Xtest, labelsInfoTest = load_train_test_data()

    xtrain, xtest, ytrain, ytest = train_test_split(xTrain, yTrain, test_size=0.5)

    model = KNN(n_neighbors=60)
    model.fit(xtrain, ytrain)
    print model.score(xtest, ytest)
    ytest_pred = model.predict(xtest)
    print accuracy_score(ytest_pred, ytest)

def test_knn_model():
    xTrain, yTrain, Xtest, labelsInfoTest = load_train_test_data()

    xtrain, xtest, ytrain, ytest = train_test_split(xTrain, yTrain, test_size=0.5)
    tuned_parameters = [{"n_neighbors":list(range(1,5))}]
    clf = GridSearchCV( model, tuned_parameters, cv=5, scoring="accuracy")
    clf.fit(xtrain, ytrain)
    print clf.grid_scores_

def get_submission():
    xTrain, yTrain, xTest, labelsInfoTest = load_train_test_data()
   
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    model.fit(xTrain, yTrain)
    yTest = model.predict(xTest)
    
    print labelsInfoTest.shape, yTest.shape
    
    yTest2 = map(transform_feature_to_str, yTest)
    
    submit_df = labelsInfoTest
    submit_df['Class'] = yTest2
    submit_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    #load_train_test_data()
    #train_nn_model()
    train_knn_model()
    #train_model()
    #get_submission()
