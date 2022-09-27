# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:51:19 2022

@author: Kelly Johnson
windows 10, tensorflow 2.3, vs 2019, cuda 10.1, cudnn 7.6
10x10 multiclass classification attempt adapted from Jason Brownlee's 
"Multi-Class Classification Tutorial with Keras Deep Learning Library
https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
"""


# multi-class classification with Keras
import numpy as np
import random
import time
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xlsxwriter
from datetime import datetime

# check gpu config and see if a minimum of 7 compute power
# =============================================================================
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print(tf.test.is_built_with_cuda) 
# tf.config.list_physical_devices('GPU')
# gpu_available = tf.test.is_gpu_available()
# is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
# is_cuda_gpu_min_7 = tf.test.is_gpu_available(True, (7,0))
# print(is_cuda_gpu_min_7)
# =============================================================================


# input data
# INPUT_DATA_PATH = "norm_tenByTenModelData.csv"

# initiate clock
# =============================================================================
# TODAYS_DATETIME = datetime.now().strftime("%m-%d-%Y_%H%M%S")
# TODAYS_DATE = datetime.now().strftime("%m-%d-%Y")
# TODAYS_TIME = datetime.now().strftime("%H:%M:%S %Z")
# =============================================================================
# load dataset
dataframe = pd.read_csv("tenByTenModelData.csv")
dataset = dataframe.values
X = dataset[:,0:3].astype(float)
Y = dataset[:,3]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
 
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(512, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(128, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(64, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(100, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
estimator = KerasClassifier(build_fn=baseline_model, epochs=2500, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline Accuracy (Standard Deviation): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
