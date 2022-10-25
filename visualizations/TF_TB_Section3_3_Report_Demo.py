# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:29:20 2022

@author: Kelly Johnson
"""


import tensorflow as tf
import pandas as pd
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard as TB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

sc = StandardScaler()

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

USER = 'Kelly'
INPUT_DATA_PATH = "norm_tenByTenModelData.csv"
#DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


# Model Parameters
ACTIVATION_FUNC = 'relu'
OUT_ACTIVATION_FUNC = 'softmax'
KERNEL_INITIALIZER = 'he_uniform'
HIDDEN_LAYER_NODE_1 = 512
HIDDEN_LAYER_NODE_2 = 256
HIDDEN_LAYER_NODE_3 = 128
HIDDEN_LAYER_NODE_4 = 64

MAX_EPOCHS = 2500
EP_LOG_INTERVAL = MAX_EPOCHS / 4
BATCH_SIZE = 40
LEARNING_RATE = 0.02
#In Model Compile...Choose which one: Adam, SGD, Adagrad 
LOSS_FUNCTION = SparseCategoricalCrossentropy

VALIDATION_SPLIT = 0.2
X_NORMALIZE_FACTOR = 10
NUM_FEATURES = 3
NUM_CLASSES = 100
NUM_CASES = 800

# =============================================================================
log_dir = "logs/fit/adam/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TB(log_dir=log_dir)
# To access tensorboard type from command line:
# Go to the python environment and type: tensorboard --logdir logs/fit 
# =============================================================================

# load data
data = pd.read_csv(INPUT_DATA_PATH)
dataset = data.values

X = dataset[:,0:3].astype(float)
Y = dataset[:,3]

#One Hot Encoded class values for Categorical Crossentropy
# =============================================================================
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = to_categorical(encoded_Y)
# =============================================================================


##If using Categorical Crossentropy, Change Y to dummy_y
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = VALIDATION_SPLIT)

# create model
def create_model():
    return Sequential([
        Dense(HIDDEN_LAYER_NODE_1, input_dim = NUM_FEATURES, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
        Dense(HIDDEN_LAYER_NODE_2, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
        Dense(HIDDEN_LAYER_NODE_3, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
        Dense(HIDDEN_LAYER_NODE_4, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
        Dense(NUM_CLASSES, kernel_initializer = KERNEL_INITIALIZER, activation = OUT_ACTIVATION_FUNC)
    ])
model = create_model()

# =============================================================================
# In model.compile:
# To change optimizer: Change Adam to SGD or Adagrad
# To Change loss: Change SparseCategoricalCrossentropy to CategoricalCrossentropy
# =============================================================================
model.compile(
    optimizer=Adam(learning_rate=0.02),
    loss=SparseCategoricalCrossentropy(), 
    metrics=['accuracy']
    )

model.fit(x=X_train, 
          y=Y_train, 
          epochs=MAX_EPOCHS, 
          batch_size = BATCH_SIZE,
          validation_data=(X_test, Y_test), 
          callbacks=[tensorboard_callback])
