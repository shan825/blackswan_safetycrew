""
Created on Tue Sep  6 15:41:41 2022

@author: reiva
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sc = StandardScaler()

""" Experiment 1.1 : Complete dataset for training and testing"""

data = pd.read_csv('data690_last.csv')

m = len(data)
## Splitting Dataset ###
X = data.iloc[:,:2].values
Y = data.iloc[:,-1].values

# Splitting dataset into training and testing dataset


X_train, X_test_un, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0, stratify=Y)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test_un)
ann = tf.keras.models.Sequential()

# five layer ANN
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))

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


Actual_pred_test_df = pd.DataFrame(X_test_un,columns = ["objixPos", "objiDir"])
Actual_pred_test_df = pd.concat([Actual_pred_test_df,pred_test_df], axis = 1)
#print(Actual_pred_test_df.head())
Actual_pred_test_df.to_csv('C:/Users/reiva/Exp1.1_fulldataset+pred.csv', index= False)


""" Experiment 1.2 : 80% initial conditions for training """

training_data = pd.read_csv('C:/Users/reiva/Exp12_traindataset.csv')
testing_data = pd.read_csv('C:/Users/reiva/Exp1.2_testdataset.csv')
fulldata_test = pd.read_csv('C:/Users/reiva/Exp1.2_fulldata_test.csv')

### training set ####
X = training_data.iloc[:,:2].values
Y = training_data.iloc[:,-2].values

### testing set ####
x = testing_data.iloc[:,:2].values
y = testing_data.iloc[:,-2].values

## full data for testing ##
X_full_test_un = fulldata_test.iloc[:,:2].values
Y_full_test = fulldata_test.iloc[:,-2].values

######assigning train and test #####
X_train, X_test_un, Y_train, Y_test = X,x,Y,y
X_test_dup = X_test_un
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test_un)
X_full_test = sc.transform(X_full_test_un)

#print(Y_test.shape[0],Y_train.shape[0])
ann = tf.keras.models.Sequential()

# Single layer ANN
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))
ann.add(tf.keras.layers.Dense(20, input_dim= 2, kernel_initializer='he_uniform', activation='relu'))

ann.add(tf.keras.layers.Dense(1))

ann.compile(optimizer="adam", loss='mae', metrics=['accuracy'])
ann.fit(X_train, Y_train, epochs=100, verbose=0)
test_loss, test_accuracy = ann.evaluate(X_test, Y_test)
print("Exp1, test loss, test accuracy",test_loss,test_accuracy)

Y_pred = ann.predict(X_test)
Y_full_pred = ann.predict(X_full_test)

#Y_pred

Y_pred = np.round(Y_pred, 0)
Y_pred = np.round(abs(Y_pred))
Y_pred = pd.DataFrame(Y_pred)
#Y_pred_full

Y_full_pred = np.round(Y_full_pred, 0)
Y_full_pred = np.round(abs(Y_full_pred))
Y_full_pred = pd.DataFrame(Y_full_pred)


#print(Y_pred)
Y_test = pd.DataFrame(Y_test)
Y_full_test = pd.DataFrame(Y_full_test)


pred_test_df = pd.concat([Y_pred, Y_test],axis=1)
pred_test_df.columns=['Y_pred','Y_test']
#pred_test_df.to_csv('./output/Exp1.2_predictedVSactual.csv', index= False)

pred_full_df = pd.concat([Y_full_pred, Y_full_test],axis=1)
pred_full_df.columns=['Y_full_pred','Y_test']

Actual_pred_test_df = pd.DataFrame(X_test_un,columns = ["objixPos", "objiDir"])
Actual_pred_test_df = pd.concat([Actual_pred_test_df,pred_test_df], axis = 1)
#print(Actual_pred_test_df.head())
Actual_pred_test_df.to_csv('C:/Users/reiva/Exp1.2_dataset+pred_70_30_4cn_2pn.csv', index= False)


#### exporting full data set testing
complete_pred_test_df = pd.DataFrame(X_full_test_un,columns = ["objixPos", "objiDir"])
complete_pred_test_df = pd.concat([complete_pred_test_df,pred_full_df], axis = 1)
#print(Actual_pred_test_df.head())
complete_pred_test_df.to_csv('C:/Users/reiva/Exp1.2_completedatatest+pred_70_30_4cn_2pn.csv', index= False)



""" Calculationg Accuracy """
correct_prediction = 0
incorrect_prediction = 0


pred_test_df['correct_prediction'] = np.where(pred_test_df.iloc[:,0] == pred_test_df.iloc[:,1], 1, 0)
pred_full_df['correct_prediction'] = np.where(pred_full_df.iloc[:,0] == pred_full_df.iloc[:,1], 1, 0)


accuracy_hiddenset = pred_test_df['correct_prediction'].sum()/len(pred_test_df['correct_prediction'])
accuracy_fullset = pred_full_df['correct_prediction'].sum()/len(pred_full_df['correct_prediction'])

print('test Accuracy: %f',accuracy_hiddenset, accuracy_fullset) #, acc_val)
