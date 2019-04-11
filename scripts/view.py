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

CLASS_NUM=4
CLASS_NAME=['v_short','short','medium','long']

CLASS_NUM_GENDER=2
CLASS_NAME_GENDER=['mens','women']


def load_model():
    model_womens = model_from_json(open('../models/model_predict/womens/model.json').read())
    model_womens.load_weights('../models/model_predict/womens/weights.h5')
    model_womens.summary()
    model_womens.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 0.0001),
                  metrics = ['accuracy'])
    model_mens = model_from_json(open('../models/model_predict/mens/model.json').read())
    model_mens.load_weights('../models/model_predict/mens/weights.h5')
    model_mens.summary()
    model_mens.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 0.0001),
                  metrics = ['accuracy'])
    model_gender = model_from_json(open('../models/model_predict/gender/model.json').read())
    model_gender.load_weights('../models/model_predict/gender/weights.h5')
    model_gender.summary()
    model_gender.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 0.0001),
                  metrics = ['accuracy'])

    return model_mens,model_womens,model_gender

def predict_gender(image):#性別予想
	p=model_gender.predict(image)
	p=p.reshape(CLASS_NUM_GENDER)
	print(p)
	label=0
	for i in range(len(p)):
		if p[i]>=0.5:
			label=i
#	print(CLASS_NAME_GENDER[label])
	return label

def img_regularize(image):#切り抜き正規化
	img_tru=image[210:510,490:790,:]
	img_tru=np.array(img_tru)
	img_tru=cv2.resize(img_tru,(200,200))
	img_tru=img_tru.reshape(1,200,200,3)
	img_tru=img_tru.astype('float32')
	img_tru/=255	
	return img_tru

def suggest_style(frame_deal,set_gen=-1):#髪型提案
	#webカメラからデータ整形
	img_tru=img_regularize(frame_deal)

	#gender予想
	if set_gen==-1:
		gender=predict_gender(img_tru)
	else:
		gender=set_gen

	ans=0

	#髪型予想
	if gender==0:#男
		ans=model_mens.predict(img_tru)
		ans=ans.reshape(CLASS_NUM)

	elif gender==1:#女
		ans=model_womens.predict(img_tru)
		ans=ans.reshape(CLASS_NUM)

	print(ans)
	label=0
	for i in range(len(ans)):
		if ans[i]==1:
			label=i

	print(gender)
	print(CLASS_NAME[label])
	
	if gender==0:#男
		frame_deal=frame_deal.astype('float32')
		frame_deal[0:720:, 0:1280] *= 1 - hair_mask_mens[label]  # 透過率に応じて元の画像を暗くする。
		frame_deal[0:720:, 0:1280] += hair_mens[label] * hair_mask_mens[label]  # 貼り付ける方の画像に透過率をかけて加算。
		frame_deal=frame_deal.astype('uint8')

	elif gender==1:#女
		frame_deal=frame_deal.astype('float32')
		frame_deal[0:720:, 0:1280] *= 1 - hair_mask_women[label]  # 透過率に応じて元の画像を暗くする。
		frame_deal[0:720:, 0:1280] += hair_women[label] * hair_mask_women[label]  # 貼り付ける方の画像に透過率をかけて加算。
		frame_deal=frame_deal.astype('uint8')

	cv2.imshow('suggested style', frame_deal)

	tdatetime = dt.now()
	tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
	tstr = "../save_images/" + tstr
	cv2.imwrite(tstr+'_raw.png',frame)
	cv2.imwrite(tstr+'_suggested.png',frame_deal)


	cv2.waitKey()
	cv2.destroyWindow('suggested style')

def suggest_ttm(frame_deal,set_gen=1):#髪型提案
	#webカメラからデータ整形
	img_tru=img_regularize(frame_deal)

	#gender予想
	if set_gen==-1:
		gender=predict_gender(img_tru)
	else:
		gender=set_gen

	ans=0

	label=2
	print(gender)
	print(CLASS_NAME[label])
	
	if gender==0:#男
		frame_deal=frame_deal.astype('float32')
		frame_deal[0:720:, 0:1280] *= 1 - hair_mask_mens[label]  # 透過率に応じて元の画像を暗くする。
		frame_deal[0:720:, 0:1280] += hair_mens[label] * hair_mask_mens[label]  # 貼り付ける方の画像に透過率をかけて加算。
		frame_deal=frame_deal.astype('uint8')

	elif gender==1:#女
		frame_deal=frame_deal.astype('float32')
		frame_deal[0:720:, 0:1280] *= 1 - hair_mask_women[label]  # 透過率に応じて元の画像を暗くする。
		frame_deal[0:720:, 0:1280] += hair_women[label] * hair_mask_women[label]  # 貼り付ける方の画像に透過率をかけて加算。
		frame_deal=frame_deal.astype('uint8')

	cv2.imshow('suggested style', frame_deal)

	tdatetime = dt.now()
	tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
	tstr = "../save_images/" + tstr
	cv2.imwrite(tstr+'_raw.png',frame)
	cv2.imwrite(tstr+'_suggested.png',frame_deal)


	cv2.waitKey()
	cv2.destroyWindow('suggested style')	

def load_alpha(directory):
	src = cv2.imread(directory, -1)  # -1を付けることでアルファチャンネルも読んでくれるらしい。

	width, height = src.shape[:2]

	mask = src[:,:,3]  # アルファチャンネルだけ抜き出す。
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # 3色分に増やす。
	mask.astype('float32')
	mask = mask / 255.0  # 0-255だと使い勝手が悪いので、0.0-1.0に変更。

	src = src[:,:,:3]  # アルファチャンネルは取り出しちゃったのでもういらない。
	return src,mask

#モデル読み込み
model_mens,model_womens,model_gender=load_model()
cap = cv2.VideoCapture(0)
face,face_mask=load_alpha('../images/face.png')
hair_women=[]
hair_mask_women=[]
hair_mens=[]
hair_mask_mens=[]

for i in range(CLASS_NUM):
	img,img_mask=load_alpha('../images/women_'+CLASS_NAME[i]+'.png')
	hair_women.append(img)
	hair_mask_women.append(img_mask)

for i in range(CLASS_NUM):
	img,img_mask=load_alpha('../images/mens_'+CLASS_NAME[i]+'.png')
	hair_mens.append(img)
	hair_mask_mens.append(img_mask)

if __name__=='__main__':
	while True:
		ret, frame = cap.read()
		frame=cv2.flip(frame,1)
		frame_copy=np.copy(frame)
		frame_copy=frame_copy.astype('float32')
		frame_copy[0:720:, 0:1280] *= 1 - face_mask  # 透過率に応じて元の画像を暗くする。
		frame_copy[0:720:, 0:1280] += face * face_mask  # 貼り付ける方の画像に透過率をかけて加算。
		frame_copy=frame_copy.astype('uint8')

		#frameを表示
		cv2.imshow('camera capture', frame_copy)
		#10msecキー入力待ち
		k = cv2.waitKey(10)
		#Escキーを押されたら終了
		print(k)
		if k == 27:#ESCでオワオワリ
			break
		elif k== 100:#Dで撮影
			suggest_style(frame,-1)
		elif k==115:#画面キャプチャ
			cv2.imwrite('face.png',frame_copy)
		elif k==109:#Mで男認識
			suggest_style(frame,0)
		elif k==119:#Wで女認識
			suggest_style(frame,1)
		elif k==116:#Tで堤
			suggest_ttm(frame,1)
	#キャプチャを終了
	cap.release()
	cv2.destroyAllWindows()