#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as pl

import gzip

import pandas as pd

from load_fn import load_data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

from sklearn.feature_selection import RFECV

from sklearn.cross_validation import cross_val_score as k_fold_CV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA, FastICA, KernelPCA

from sklearn.pipeline import Pipeline

import cPickle as pickle

ORD_VALUES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
NORD = len(ORD_VALUES)

def transform_str_to_feature(st):
    return ORD_VALUES.index(ord(st))
    #ordval = ord(st)
    #ordidx = ORD_VALUES.index(ordval)
    #return (ordidx-NORD//2)/float(NORD//2)

def transform_feature_to_str(ft):
    return chr(ORD_VALUES[int(ft)])
    #idx = NORD//2 + int(ft * NORD//2)
    #if idx < 0 :
        #idx = 0
    #if idx >= NORD:
        #idx = ORD_VALUES-1
    #ordval = ORD_VALUES[idx]
    #return chr(ordval)

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

def load_train_test_data(nn_ytrain=False):
    imageSize = 400 # 20 x 20 pixels

    #Set location of data files , folders
    path = '.'

    labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(path))

    #Read training matrix
    xTrain = load_data("train", labelsInfoTrain, imageSize, path)

    if nn_ytrain:
        yTrain = transform_from_classes(labelsInfoTrain['Class'])
    else:
        yTrain = labelsInfoTrain['Class'].map(transform_str_to_feature)

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

    xTrain, yTrain, xTest, labelsInfoTest = load_train_test_data(nn_ytrain=True)

    print xTrain.shape, yTrain.shape
    print xTrain.dtype, yTrain.dtype

    model.fit(xTrain, yTrain)
    ytest_pred = model.predict(xTrain)
    print model.score(xTrain, yTrain)
    print accuracy_score(ytest_pred, yTrain)
    
    yTest = model.predict(xTest)
    
    print labelsInfoTest.shape, yTest.shape
    
    yTest2 = transform_to_class(yTest)
    
    submit_df = labelsInfoTest
    submit_df['Class'] = yTest2
    submit_df.to_csv('submission.csv', index=False)

    return model

def train_model():
    xTrain, yTrain, Xtest, labelsInfoTest = load_train_test_data(nn_ytrain=True)

    xtrain, xtest, ytrain, ytest = train_test_split(xTrain, yTrain, test_size=0.5)

    for name, model in (('rf400', RandomForestClassifier(n_estimators=400, n_jobs=-1)),
                        ('knn', KNeighborsClassifier()),
                        ('knn62', KNeighborsClassifier(n_neighbors=62)),
                        ('sgdc_hinge', SGDClassifier(loss='hinge', n_jobs=-1)),
                        ('sgdc_log', SGDClassifier(loss='log', n_jobs=-1)),
                        ('sgdc_modhub', SGDClassifier(loss='modified_huber', n_jobs=-1)),
                        ('sgdc_sqhinge', SGDClassifier(loss='squared_hinge', n_jobs=-1)),
                        ('sgdc_perceptron', SGDClassifier(loss='perceptron', n_jobs=-1)),):
        model.fit(xtrain, ytrain)
        print name, model.score(xtest, ytest)
        ytest_pred = model.predict(xtest)
        print name, accuracy_score(ytest_pred, ytest)
        
        with gzip.open('%s_model.pkl.gz', 'w') as f:
            pickle.dump(model, f)

def test_knn_model():
    xTrain, yTrain, Xtest, labelsInfoTest = load_train_test_data()

    xtrain, xtest, ytrain, ytest = train_test_split(xTrain, yTrain, test_size=0.5)
    
    model = KNeighborsClassifier()
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
    #train_knn_model()
    #test_knn_model()
    train_model()
    #get_submission()
