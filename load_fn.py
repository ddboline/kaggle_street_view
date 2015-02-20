#!/usr/bin/python

import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from skimage.io import imread

FTRAIN = 'training.csv'
FTEST = 'test.csv'

def float32(k):
    return np.cast['float32'](k)

def load_data(typeData, labelsInfo, imageSize, path):
    x = np.zeros((labelsInfo.shape[0], imageSize))
    for (index, idImage) in enumerate(labelsInfo["ID"]):
        nameFile = "{0}/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
        img = imread(nameFile, as_grey=True)
        img = float32(img)
        x[index, :] = np.reshape(img, (1, imageSize))
    return x
