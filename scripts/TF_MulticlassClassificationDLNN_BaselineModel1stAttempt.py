# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 23:01:39 2022

@author: Kelly Johnson
10x10 multiclass classification attempt adapted from Jason Brownlee's 
"Multi-Class Classification Tutorial with Keras Deep Learning Library
https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
"""


# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("tenByTenModelData.csv")
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
	model.add(Dense(20, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(20, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(20, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(20, input_dim=3,kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(100, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline Accuracy (Standard Deviation): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
