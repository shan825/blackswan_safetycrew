import tensorflow
print(tensorflow.__version__)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sc = StandardScaler()
import statistics

from keras.models import Sequential
from keras.layers import Dense

#Read in Data
df = pd.read_csv(r'C:\Users\Shing\Documents\pytorch_test\data\tenByTenModelData.csv')
print(df)
# Split into Train Test Split
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]


tsize = .4
X_train, X_test_a, Y_train, Y_test = train_test_split(X, y, random_state=42,test_size=tsize)

X_test_duplicate = X_test_a
X_train = sc.fit_transform(X_train)
x_test = sc.fit_transform(X_test_a)


# Build Neural Network model
ann = tensorflow.keras.models.Sequential()
ann.add(tensorflow.keras.layers.Dense(input_dim = 2, kernel_initializer='he_uniform',activation='relu',units=100))
ann.add(tensorflow.keras.layers.Dense(input_dim = 2, kernel_initializer='he_uniform',activation='relu',units=100))
ann.add(tensorflow.keras.layers.Dense(input_dim = 2, kernel_initializer='he_uniform',activation='relu',units=100))
ann.add(tensorflow.keras.layers.Dense(1))
ann.compile(optimizer='adam',loss='mae',metrics=['accuracy'])

#TRAIN MODEL AND RUN
ann.fit(X_train, Y_train)