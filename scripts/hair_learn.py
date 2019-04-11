#coding: UTF-8
from __future__ import print_function

import cv2
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from keras.utils import np_utils
from keras.models import Sequential,Model,model_from_json
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,merge
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D,Conv2D,Conv2DTranspose
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras import backend as K
from datetime import datetime as dt
import tensorflow as tf
from keras.backend import tensorflow_backend

import cv2
import numpy as np
import os

import random


CLASS_NUM=5
CLASS_NAME=['v_short','short','medium','long','v_long']

# ルックアップテーブルの生成
min_table = 50
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256,-1,-1, dtype = 'uint8' )
LUT_LC = np.arange(256,-1,-1, dtype = 'uint8' )

# ハイコントラストLUT作成
for i in range(0, min_table):
	LUT_HC[i] = 0
for i in range(min_table, max_table):
	LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 256):
	LUT_HC[i] = 255

LUT_HC=cv2.merge((LUT_HC,LUT_HC,LUT_HC))

	# ローコントラストLUT作成
for i in range(256):
	LUT_LC[i] = min_table + i * (diff_table) / 255
LUT_LC=cv2.merge((LUT_LC,LUT_LC,LUT_LC))

def model_generate():
	model = Sequential()
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv) ,activation='relu',padding = 'same', input_shape = x_train.shape[1:]))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.20))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.20))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.20))	
	model.add(Flatten())
	model.add(Dense(100,activation='relu'))
	model.add(Dense(50,activation='relu'))
	model.add(Dense(CLASS_NUM, activation='softmax'))
	model.summary()
	model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = LearningRate),
                  metrics = ['accuracy'])
	return model

#評価や学習用データ生成
def load_data(directory, batch_size):
	file_dist=[random.random(),random.random(),random.random(),random.random(),random.random()]
	s=sum(file_dist)
#	print(file_dist)
	print(s)
	for i in range(len(file_dist)):
		file_dist[i]/=s
		file_dist[i]*=batch_size
		file_dist[i]=int(file_dist[i])
	dif=batch_size-sum(file_dist)
	file_dist[random.randrange(CLASS_NUM)]+=dif
	print(file_dist)
	X=[]
	Y=[]
	output=[]
	for i in range(CLASS_NUM):
		file_list=os.listdir(directory+CLASS_NAME[i])
#		print(file_list)
		batch_list = random.sample(range(len(file_list) - 1), file_dist[i])
		for batch in batch_list:
			img=cv2.imread(directory+CLASS_NAME[i]+'/'+str(batch)+'.png')
			data_augumentation(img)
			img=np.array(img)
			X.append(img)
			output.append(i)
	X = np.array(X)
	X = X.astype('float32')
	X /= 255
	Y=keras.utils.to_categorical(output,CLASS_NUM)
#	print(X)
#	print(Y)
	return X,Y

def data_augumentation(img):
#	if np.random.rand()<0.5:
		# チャンネル分解
#		for i in range(200):
#			for j in range(200):
#				temp=0.333333 * img[j,i,0]+0.333333 * img[j,i,1]+0.333333 *  img[j,i,2]
#				img[j,i,0]=temp
#				img[j,i,1]=temp
#				img[j,i,2]=temp
#		r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
#		r_r=np.array([r,r,r]).astype('float32')
#		r_g=np.array([g,g,g]).astype('float32')
#		r_b=np.array([b,b,b]).astype('float32')
#		src = 0.2989 * r_r + 0.5870 * r_g + 0.1140 * r_b
#		img=np.array(src)
#		print('success')
#		print(img.shape)
#		temp=np.ndarray();
#		temp=np.append()
#		img=np.append(img,src)
#		img=np.append(img,src)
#	if np.random.rand()<0.5:
#		print(img.shape)
#		img=img[:, ::-1,:]
	if np.random.rand()<0.5:
		row,col,ch= img.shape
		mean = 0
		sigma = 10
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		img = img + gauss

# 変換
#	t=np.random.rand()
#	if t<0.3333:
#		print(LUT_HC)
#		img = cv2.LUT(img, LUT_HC)
#	elif t<0.6666:
#		print(LUT_LC)
#		img = cv2.LUT(img, LUT_LC)

def save_model(model, name_model, name_weights):
    model_json_str = model.to_json()
    open(tstr + name_model, 'w').write(model_json_str)
    model.save_weights(tstr + name_weights)
    print(name_model)
    print(name_weights)
    print("saved")



if __name__ == '__main__':
	config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
	session = tf.Session(config = config)
	tensorflow_backend.set_session(session)

	#各種のパラメータ
	nb_filters = 128
	nb_conv = 4
	nb_pool = 2
	nb_epoch = 200
	batch_sample = 16
	nb_iter = int(2400*8 / batch_sample)
	LearningRate = 0.001
	#モデルを保存するフォルダ名
	tdatetime = dt.now()
	tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
	tstr = "../models/model_" + tstr + "/"
	os.makedirs(tstr)
	train_dir = "../datasets/"
	#学習用データとテスト用データ
	x_train = []
	y_train = []
	x_train, y_train = load_data(train_dir, batch_sample)
	print(x_train.shape[1:])
	model = model_generate()

	for i in range(nb_epoch):
		if i%10 == 0:
			LearningRate *= 0.8
			K.set_value(model.optimizer.lr, LearningRate)
			save_model(model, 'model'+str(i)+'.json', 'weights'+str(i)+'.h5')
		for j in range(nb_iter):
			x_train, y_train = load_data(train_dir, batch_sample)
			model.fit(x_train, y_train, batch_size = batch_sample, epochs = 1, verbose = 1, validation_split = 0.1)
	save_model(model, 'model_final.json', 'weights_final.h5')
