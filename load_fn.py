#!/usr/bin/python

import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from skimage.io import imread

FTRAIN = 'training.csv'
FTEST = 'test.csv'

def load_data(typeData, labelsInfo, path):
    x = np.zeros((labelsInfo.shape[0], 20, 20))
    for (index, idImage) in enumerate(labelsInfo["ID"]):
        nameFile = "{0}/{1}Resized/{2}.Bmp".format(path, typeData, idImage)
        img = imread(nameFile, as_grey=True).astype(np.float32)
        x[index, :, :] = img
        #np.reshape(img, (1, imageSize))
    return x
