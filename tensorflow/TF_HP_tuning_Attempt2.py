# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:40:04 2022

@author: Kelly Johnson

https://keras.io/keras_tuner/
https://ml-course.github.io/master/labs/Lab%206%20-%20Tutorial#predictions-and-evaluations
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
from tensorflow.keras import Sequential, optimizers, callbacks
from tensorflow.keras.layers import Dense, Activation
from keras_tuner.tuners import RandomSearch
from IPython.display import clear_output

# =============================================================================
# For plotting the learning curve in real time
# =============================================================================
class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        self.max_acc = 0
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.max_acc = max(self.max_acc, logs.get('val_accuracy'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure(figsize=(8,3))
            plt.plot(N, self.losses, lw=2, c="b", linestyle="-", label = "train_loss")
            plt.plot(N, self.acc, lw=2, c="r", linestyle="-", label = "train_acc")
            plt.plot(N, self.val_losses, lw=2, c="b", linestyle=":", label = "val_loss")
            plt.plot(N, self.val_acc, lw=2, c="r", linestyle=":", label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}, Max Acc {:.4f}]".format(epoch, self.max_acc))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            #plt.show()

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

def build_model(hp):
  model = Sequential()
  # Tune the number of units in the dense layers
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)

  model.add(Dense(units = hp_units, activation = 'relu', kernel_initializer='he_uniform',input_shape=(3,)))
  model.add(Dense(units = hp_units, activation = 'relu', kernel_initializer='he_uniform'))
  model.add(Dense(units = hp_units, activation = 'relu', kernel_initializer='he_uniform'))
  model.add(Dense(100, activation='softmax'))
 
  # Tune the learning rate for the optimizer 
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 

  ###COMMENT OUT THE OPTIMIZER YOU DON'T WANT TO USE
  model.compile(optimizer = optimizers.Adam(learning_rate = hp_learning_rate),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
  
  #model.compile(optimizer = optimizers.SGD(learning_rate = hp_learning_rate),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
  return model

# =============================================================================
# Initialize a tuner (here, RandomSearch). We use objective to 
# specify the objective to select the best models, and we use 
# max_trials to specify the number of different models to try.
# Change project_name each time you run
# =============================================================================

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    project_name='bssc')

# =============================================================================
# Start the search and get the best model:
# =============================================================================

# Uncomment to run. It takes a while.
tuner.search(Xf_train, yf_train, epochs = 1000, validation_data = (x_val, y_val), callbacks = [TrainingPlot()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
