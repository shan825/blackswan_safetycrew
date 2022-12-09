# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:07:49 2022

@author: Kelly Johnson
https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
"""

# mlp for multiclass classification
import numpy as np
from numpy import argmax
from pandas import read_csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# load the dataset
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
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]

# This returns a multi-layer-perceptron model in Keras.
def get_keras_model(num_hidden_layers, 
                    num_neurons_per_layer, 
                    dropout_rate, 
                    activation):
    # create the MLP model.
    
    # define the layers.
    inputs = tf.keras.Input(shape=(X_train.shape[1],))  # input layer.
    x = layers.Dropout(dropout_rate)(inputs) # dropout on the weights.
    
    # Add the hidden layers.
    for i in range(num_hidden_layers):
        x = layers.Dense(num_neurons_per_layer, 
                         activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # output layer.
    outputs = layers.Dense(100, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
    

# This function takes in the hyperparameters and returns a score (Cross validation).
def keras_mlp_cv_score(parameterization, weight=None):
    
    model = get_keras_model(parameterization.get('num_hidden_layers'),
                            parameterization.get('neurons_per_layer'),
                            parameterization.get('dropout_rate'),
                            parameterization.get('activation'))
    
    opt = parameterization.get('optimizer')
    opt = opt.lower()
    
    learning_rate = parameterization.get('learning_rate')
    
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt == 'rms':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    NUM_EPOCHS = 50
    
    # Specify the training configuration.
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    data = X_train
    labels = y_train
    
    # fit the model using a 20% validation set.
    res = model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=parameterization.get('batch_size'),
                    validation_split=0.2)
    
    # look at the last 10 epochs. Get the mean and standard deviation of the validation score.
    last10_scores = np.array(res.history['val_loss'][-10:])
    mean = last10_scores.mean()
    sem = last10_scores.std()
    
    # If the model didn't converge then set a high loss.
    if np.isnan(mean):
        return 9999.0, 0.0
    
    return mean, sem
# Define the search space.
parameters=[
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [0.0001, 0.5],
        "log_scale": False,
    },
    {
        "name": "dropout_rate",
        "type": "range",
        "bounds": [0.01, 0.5],
        "log_scale": False,
    },
    {
        "name": "num_hidden_layers",
        "type": "range",
        "bounds": [1, 10],
        "value_type": "int"
    },
    {
        "name": "neurons_per_layer",
        "type": "range",
        "bounds": [1, 600],
        "value_type": "int"
    },
    {
        "name": "batch_size",
        "type": "choice",
        "values": [8, 16, 32, 64, 128, 256],
    },
    
    {
        "name": "activation",
        "type": "choice",
        "values": ['tanh', 'sigmoid', 'relu'],
    },
    {
        "name": "optimizer",
        "type": "choice",
        "values": ['adam', 'rms', 'sgd'],
    },
]

# import more packages
from ax.service.ax_client import AxClient
#from ax.utils.notebook.plotting import render, init_notebook_plotting

#init_notebook_plotting()

ax_client = AxClient()

# create the experiment.
ax_client.create_experiment(
    name="keras_experiment",
    parameters=parameters,
    objective_name='keras_cv',
    minimize=True)

def evaluate(parameters):
    return {"keras_cv": keras_mlp_cv_score(parameters)}

for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


# look at all the trials.
print(ax_client.get_trials_data_frame().sort_values('trial_index'))

best_parameters, values = ax_client.get_best_parameters()

# the best set of parameters.
for k in best_parameters.items():
  print(k)

print()

# train the model on the full training set and test.
keras_model = get_keras_model(best_parameters['num_hidden_layers'], 
                              best_parameters['neurons_per_layer'], 
                              best_parameters['dropout_rate'],
                              best_parameters['activation'])


opt = best_parameters['optimizer']
opt = opt.lower()

learning_rate = best_parameters['learning_rate']

if opt == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
elif opt == 'rms':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

NUM_EPOCHS = 50

# Specify the training configuration.
keras_model.compile(optimizer=optimizer,
              loss=tf.keras.losses.Categorical_Crossentropy(),
              metrics=['accuracy'])

data = X_train
labels = y_train.values
res = keras_model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=best_parameters['batch_size'])

# Use the model to predict the test values.
test_pred = keras_model.predict(X_test)
print(accuracy(y_test, test_pred))
# =============================================================================
# 
# # define model
# model = Sequential()
# model.add(Dense(512, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
# model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
# model.add(Dense(100, activation='softmax'))
# # compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # fit the model
# model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=2)
# # evaluate the model
# loss, acc = model.evaluate(X_test, y_test, verbose=2)
# print('Test Accuracy: %.3f' % acc)
# =============================================================================
# make a prediction
# =============================================================================
# row = [5.1,3.5,1.4]
# yhat = model.predict([row])
# print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# =============================================================================
