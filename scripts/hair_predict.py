#coding: UTF-8
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from datetime import datetime as dt

import cv2
import numpy as np
import os

import random

CLASS_NUM=5
CLASS_NAME=['v_short','short','medium','long','v_long']

def load_data(directory):
    caltech_dir = directory
    file_list = os.listdir(caltech_dir)
    file_num = len(file_list)
    print(file_list)

    X = [] #インプット
    filename_list = []

    for file_path in file_list:
        from_path = caltech_dir + "/" + file_path
        img_input = cv2.imread(from_path)
        img_input = np.array(img_input)
        X.append(img_input)
        filename_list.append(file_path)

    X = np.array(X)
    X = X.astype('float32')
    X /= 255
    filename_list = np.array(filename_list)

    print(X.shape)
    return X, filename_list

def o_h_decode(arr):
    ret=np.arange(len(arr))
    for i,vec in zip(range(len(arr)),arr):
        for j in range(len(vec)):
            if vec[j]==1:
                ret[i]=j
                break
    return ret


if __name__ == '__main__':

    #各種パラメータ設定
    image_input_w = 200
    image_input_h = 200

    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
    tstr = "../datasets/predict/predict_" + tstr + "/"
#    os.makedirs(tstr)

    #学習結果を読み込む
    model = model_from_json(open('../models/model_predict/model.json').read())
    model.load_weights('../models/model_predict/weights.h5')
    model.summary()

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 0.0001),
                  metrics = ['accuracy'])
    #予測したいデータ準備
    input_data, file_list = load_data("../datasets/predict")
    arr = model.predict(input_data)
    label=o_h_decode(arr)
#    print(o_h_decode(arr))
    for i in range(len(arr)):
        print(file_list[i]+':'+CLASS_NAME[label[i]])
