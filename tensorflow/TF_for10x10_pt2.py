# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:41:41 2022

@author: reiva
editing Sp22 code from 1x10 to 10x10
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sc = StandardScaler()

""" Experiment 1.1 : Complete dataset for training and testing"""

data = pd.read_csv('tenByTenModelData.csv')

m = len(data)
print(m)
## Splitting Dataset ###
X = data.iloc[:,:3].values
Y = data.iloc[:,-1].values
np.shape(X)
np.shape(Y)

# Splitting dataset into training and testing dataset


X_train, X_test_un, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0, stratify=Y)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test_un)
ann = tf.keras.models.Sequential()

# five layer ANN
ann.add(tf.keras.layers.Dense(20, input_dim= 3, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 3, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 3, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 3, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 3, kernel_initializer='he_uniform', activation='relu'))

ann.add(tf.keras.layers.Dense(1))

ann.compile(optimizer="adam", loss='mae', metrics=['accuracy'])
model = ann.fit(X_train, Y_train, epochs=100, verbose=0)


Y_pred = ann.predict(X_test)
#Y_pred

Y_pred = np.round(Y_pred, 0)
Y_pred = np.round(abs(Y_pred))
Y_pred = pd.DataFrame(Y_pred)

Y_test = pd.DataFrame(Y_test)

pred_test_df = pd.concat([Y_pred, Y_test],axis=1)
pred_test_df.columns=['Y_pred','Y_test']


Actual_pred_test_df = pd.DataFrame(X_test_un,columns = ["xTemp", "yVol","direction"])
Actual_pred_test_df = pd.concat([Actual_pred_test_df,pred_test_df], axis = 1)
##print(Actual_pred_test_df.head())
Actual_pred_test_df.to_csv('Exp1.1_fulldataset+pred.csv', index= False)

################START BACK HERE#########################
