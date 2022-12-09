# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:51:19 2022

@author: Kelly Johnson
windows 10, tensorflow 2.3, vs 2019, cuda 10.1, cudnn 7.6
10x10 multiclass classification attempt adapted from Jason Brownlee's 
"Multi-Class Classification Tutorial with Keras Deep Learning Library
https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
"""


import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
sc = StandardScaler()
import statistics

# check gpu config and capabilities
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

# load data
data = pd.read_csv("tenByTenModelData.csv")
dataset = data.values

X = dataset[:,0:3].astype(float)
Y = dataset[:,3]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

np.shape(dummy_y)
# test train split
(X_train, X_test, Y_train, Y_test) = train_test_split(X,dummy_y,random_state=42, test_size=0.3)

# normalizing data
X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)

# checking shape of data
print(X_train.shape)
print(X_test.shape)

# function to create and compile model
def createModel():

    # create model
	model = Sequential()
	model.add(Dense(512, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(128, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(64, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(100, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=createModel, epochs=2500, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline Accuracy (Standard Deviation): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

