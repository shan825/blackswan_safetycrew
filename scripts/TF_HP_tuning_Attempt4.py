# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:40:04 2022

@author: Kelly Johnson

https://keras.io/keras_tuner/
https://ml-course.github.io/master/labs/Lab%206%20-%20Tutorial#predictions-and-evaluations
talos
"""
#%matplotlib inline
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from keras_tuner.tuners import RandomSearch
from IPython.display import clear_output
import talos
from talos.model.normalizers import lr_normalizer


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
#Xf_train, x_val, yf_train, y_val = train_test_split(X_train, y_train, train_size=0.80, shuffle=True, stratify=y_train, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]


# first we have to make sure to input data and params into the function
def tfTest3(X_train, y_train, X_test, y_test, params):
    
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=n_features,
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.add(Dropout(params['dropout']))

    model.add(Dense(100, activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'],
                  metrics=['acc', talos.utils.metrics.f1score])
    
    history = model.fit(X_train, y_train, 
                        validation_data=[X_test, y_test],
                        batch_size=params['batch_size'],
                        callbacks=[talos.callbacks.TrainingPlot(metrics=['f1score'])],
                        epochs=params['epochs'],
                        verbose=0)

    return history, model
# then we can go ahead and set the parameter space
p = {'first_neuron':[512, 256],
     'hidden_layers':[3, 4],
     'batch_size': [50],
     'epochs': [3000],
     'dropout': [0],
     'kernel_initializer': ['uniform','normal','he_uniform'],
     'optimizer': ['Adam', 'Adagrad','SGD'],
     'losses': ['categorical_crossentropy'],
     'activation':['relu', 'elu'],
     'last_activation': ['softmax']}

# and run the experiment
t = talos.Scan(x=X,
               y=y,
               model=tfTest3,
               params=p,
               experiment_name='tfTest3',
               round_limit=50,
               disable_progress_bar=True)
