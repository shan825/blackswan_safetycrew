# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:58:37 2022

@author: Kelly Johnson

https://keras.io/keras_tuner/
https://ml-course.github.io/master/labs/Lab%206%20-%20Tutorial#predictions-and-evaluations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, callbacks
from tensorflow.keras.layers import Dense, Activation
from keras_tuner.tuners import RandomSearch
from IPython.display import clear_output


# =============================================================================
# Load Data
# =============================================================================

path = 'norm_tenByTenModelData.csv'
df = read_csv(path)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = tf.keras.utils.to_categorical(y, num_classes=100, dtype='float32')
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
Xf_train, x_val, yf_train, y_val = train_test_split(X_train, y_train, train_size=0.80, shuffle=True, stratify=y_train, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]

# =============================================================================
# Write a function that creates and returns a Keras model. 
# Use the hp argument to define the hyperparameters during 
# model creation.
# =============================================================================

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def build_model(var_activation='relu',var_optimizer='adam'):
    """ Uses arguments to build Keras model. """
    model = Sequential()
    model.add(layers.Dense(512),activation=var_activation, input_shape=(3,))
    model.add(layers.Dense(125,activation=var_activation))
    model.add(layers.Dense(64,activation=var_activation))
    model.add(layers.Dense(100,activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                optimizer=var_optimizer,
                metrics=["accuracy"])
    return model

# Search space
_activations=['tanh','relu','selu']
_optimizers=['sgd','adam']
_batch_size=[16,32,64]
params=dict(var_activation=_activations,
            var_optimizer=_optimizers,
            batch_size=_batch_size)

# Wrap
model = KerasClassifier(build_fn=build_model,epochs=4,batch_size=16)

# =============================================================================
# Initialize a tuner (here, RandomSearchCV). We use objective to 
# specify the objective to select the best models, and we use 
# max_trials to specify the number of different models to try.
# =============================================================================

from sklearn.model_selection import RandomizedSearchCV

# Uncomment to run. It takes a while.
rscv = RandomizedSearchCV(model, param_distributions=params, cv=3, n_iter=10, verbose=1, n_jobs=-1)
rscv_results = rscv.fit(Xf_train,yf_train)

print('Best score is: {} using {}'.format(rscv_results.best_score_, rscv_results.best_params_))
