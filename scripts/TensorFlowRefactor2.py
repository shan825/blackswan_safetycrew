# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:33:19 2022

@author: Kelly Johnson 

Based off examples:
(Doug Cady)
https://github.com/shan825/blackswan_safetycrew/blob/main/scripts/TenByTenDLNN_VSmag_loopHoldout.py
https://visualstudiomagazine.com/articles/2022/09/06/multi-class-pytorch.aspx
https://jamesmccaffrey.wordpress.com/2022/04/27/the-iris-dataset-example-with-keras-2-8-on-windows-11/
https://jamesmccaffrey.wordpress.com/2022/04/20/the-iris-dataset-example-with-pytorch-1-10-on-windows-11/

"""
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from typing import Tuple

import numpy as np
import random
import time
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import xlsxwriter
from datetime import datetime

USER = 'Kelly'

TODAYS_DATETIME = datetime.now().strftime("%m-%d-%Y_%H%M%S")
TODAYS_DATE = datetime.now().strftime("%m-%d-%Y")
TODAYS_TIME = datetime.now().strftime("%H:%M:%S %Z")

INPUT_DATA_PATH = "../data/norm_tenByTenModelData.csv"
DEVICE = tf.device("cpu:0")

# Training/Testing Parameters
RANDOM_SEED = 160
TESTING_SPLIT_PERC = 0.25

# Model Parameters
ACTIVATION_FUNC = 'relu'
KERNEL_INITIALIZER = 'he_uniform'
OUT_ACTIVATION_FUNC = 'softmax'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
HIDDEN_LAYER_NODE_1 = 512
HIDDEN_LAYER_NODE_2 = 128
HIDDEN_LAYER_NODE_3 = 64

MAX_EPOCHS = 3_000
EP_LOG_INTERVAL = MAX_EPOCHS / 4
BATCH_SIZE = 40
LEARNING_RATE = 0.001

X_NORMALIZE_FACTOR = 10
NUM_FEATURES = 3
NUM_CLASSES = 100
NUM_CASES = 800

CASE_RESULTS_FN = f"../model_output/TensorFlow_800caseResults__{TODAYS_DATETIME}.xlsx"

# =============================================================================
# # create model
# class Net(Model):
#     def __init__(self):
#         super().__init__()
#         self.sequence = Sequential([
#             Dense(HIDDEN_LAYER_NODE_1, input_shape = 3, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
#             Dense(HIDDEN_LAYER_NODE_2, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
#             Dense(HIDDEN_LAYER_NODE_3, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
#             Dense(NUM_CLASSES, kernel_initializer = KERNEL_INITIALIZER, activation = OUT_ACTIVATION_FUNC)
#         ])
# 
#     def call(self, x: tf.Tensor) -> tf.Tensor:
#         z = self.sequence(x)
#         return z
# =============================================================================
    
class Logger(keras.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss = logs.get('loss')
      curr_acc = logs.get('accuracy') * 100
      print("epoch = %4d  |  loss = %0.6f  |  acc = %0.2f%%" % \
(epoch, curr_loss, curr_acc))

def read_split_data():
   data = pd.read_csv("norm_TenByTenModelData.csv")
   print(f"[+] Read {len(data)} rows from csv file.")
    
   x_all = data.iloc[:, 0:3].to_numpy()
   y_all = data.iloc[:, 3].to_numpy()
   return x_all, y_all


def split_train_test_holdout(x_all: np.array, y_all: np.array, case_num: int):
    """Split x, y into training, testing, and holdout datasets."""
    holdout = {
        'x': x_all[case_num, :],
        'y': y_all[case_num],
    }

    x = np.delete(x_all, (case_num), axis=0)
    y = np.delete(y_all, (case_num))

    train = {}
    test = {}
    train['x'], test['x'], train['y'], test['y'] = train_test_split(
        x, y, test_size = TESTING_SPLIT_PERC, random_state = 42
    )

    return train, test, holdout

def train_model(train_ds):
    NN = Sequential([
        Dense(HIDDEN_LAYER_NODE_1, input_shape = NUM_FEATURES, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
        Dense(HIDDEN_LAYER_NODE_2, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
        Dense(HIDDEN_LAYER_NODE_3, kernel_initializer = KERNEL_INITIALIZER, activation = ACTIVATION_FUNC),
        Dense(NUM_CLASSES, kernel_initializer = KERNEL_INITIALIZER, activation = OUT_ACTIVATION_FUNC)
        ]) 
    logger = Logger(n=10)
    OPTIMIZER = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    NN.compile(loss= LOSS_FUNCTION, optimizer = OPTIMIZER, metrics=['accuracy'])
    
    NN.fit(train_ds, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, verbose=0, callbacks=[logger]) 
  
    return NN
#def calc_model_accuracy():
    
    return 
def eval_model_accuracy(NN, test_ds, holdout):
    
    acc_test = NN.evaluate(test_ds)
    # Make prediction on holdout data
    tf_holdout = tf.Tensor(np.array([holdout['x']]), dtype=np.float32)
    
    probs = NN.predict(tf_holdout)
    
    _, pred_idx = np.where(probs == np.amax(probs))
    correct_pred_idx = holdout['y'] == pred_idx[0]
    predict_res = "Correct" if correct_pred_idx else "*Incorrect*"
    
    return acc_test, pred_idx[0], correct_pred_idx

def write_model_params(xl_writer):
    """Write model parameters to Excel file."""
    model_params = [
        ['User', USER],
        ['Date', TODAYS_DATE],
        ['Time', TODAYS_TIME],
        ['RunTime', 1],
        ['Packages', 'TensorFlow'],
        ['Dataset', '10x10'],
        ['ActivationFunction', ACTIVATION_FUNC],
        ['RandomSeed', RANDOM_SEED],
        ['BatchSize', BATCH_SIZE],
        ['Epochs', MAX_EPOCHS],
        ['InitLayerWeightBias', 'No'],
        ['Layers', 4],
        ['NodesPerLayer', f"{HIDDEN_LAYER_NODE_1}, {HIDDEN_LAYER_NODE_2}, {HIDDEN_LAYER_NODE_3}"],
        ['NetworkArch', f"{NUM_FEATURES}-({HIDDEN_LAYER_NODE_1}-{HIDDEN_LAYER_NODE_2}-{HIDDEN_LAYER_NODE_3})-{NUM_CLASSES}"],
        ['LearningRate', LEARNING_RATE],
        ['TrainingPerc', f"{(1 - TESTING_SPLIT_PERC) * 100}%"],
    ]

    df_model_params = pd.DataFrame(model_params, columns=['Parameter', 'Value'])
    df_model_params.to_excel(xl_writer, sheet_name='ModelParameters', index=False)


def print_case_results(results, case_num: int, start_time: float):
    """Print results from 800 cases and write model params, results to an Excel file."""
    results_cols = [
        'CaseNum',
        'xTemp',
        'yVol',
        'direction',
        'Target',
        'Prediction',
        'CorrectPred',
        'TestAccuracy',
    ]
    df_case_results = pd.DataFrame(results, columns=results_cols)
    xl_writer = pd.ExcelWriter(CASE_RESULTS_FN, engine='xlsxwriter')
    write_model_params(xl_writer=xl_writer)

    num_correct = df_case_results['CorrectPred'].sum()
    num_incorrect = len(df_case_results) - num_correct
    perc_correct = f"{num_correct / len(df_case_results) * 100:0.2f}%"

    min_test_accuracy = f"{df_case_results['TestAccuracy'].min() * 100:0.2f}%"
    max_test_accuracy = f"{df_case_results['TestAccuracy'].max() * 100:0.2f}%"
    mean_test_accuracy = f"{np.mean(df_case_results['TestAccuracy']) * 100:0.2f}%"
    median_test_accuracy = f"{np.median(df_case_results['TestAccuracy']) * 100:0.2f}%"
    
    result_summary = [
        ['Correct Predictions', perc_correct, num_incorrect],
        ['Test Accuracy:', '', ''],
        [' + Min', min_test_accuracy, ''],
        [' + Max', max_test_accuracy, ''],
        [' + Mean', mean_test_accuracy, ''],
        [' + Median', median_test_accuracy, '']
    ]
    df_results_summary = pd.DataFrame(result_summary, columns=['Results', 'Percentage', 'Incorrect'])
    df_results_summary.to_excel(xl_writer, sheet_name='SummaryResults', index=False)

    df_case_results['TestAccuracy'] = df_case_results['TestAccuracy'].apply(lambda x: f"{x * 100:0.2f}%")
    df_case_results.to_excel(xl_writer, sheet_name='RawResults', index=False)

    hours, minutes, seconds = print_time_elapsed(start_time=start_time)
    params_ws = xl_writer.sheets['ModelParameters']
    params_ws.write_string('B5', f"{hours}h, {minutes}m, {seconds}s")

    xl_writer.save()


def print_time_elapsed(start_time: float):
    time_elapsed = time.time() - start_time
    
    if time_elapsed > 3600:
        hours = time_elapsed // 3600
        rest = time_elapsed % 3600
        minutes = rest // 60
        seconds = round(rest % 60, 2)
        print(f"\n[+] Program finished in {int(hours)}h, {int(minutes)}m, {seconds:0.2f}s")
        return hours, minutes, seconds
    else:
        minutes = time_elapsed // 60
        seconds = round(time_elapsed % 60, 2)
        print(f"\n[+] Program finished in {int(minutes)}m, {seconds:0.2f}s")
        return 0, minutes, seconds

def main() -> int:
    start_time = time.time()

    x_all, y_all = read_split_data()
    results = []

    for case_num in range(NUM_CASES):
        train, test, holdout = split_train_test_holdout(x_all=x_all, y_all=y_all, case_num=case_num)
        train_ds = tf.data.Dataset.from_tensor_slices((train['x'],train['y']))
        test_ds = tf.data.Dataset.from_tensor_slices((test['x'],test['y']))
        train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)
        test_ds = test_ds.shuffle(1000).batch(BATCH_SIZE)
        
        if case_num % 25 == 0:
            print(f"\n[+] Case # {case_num + 1:>3}: holdout {holdout['x'] * 10} | Target: {holdout['y']}")

        NN = train_model(train_ds=train_ds)
        acc_test, pred_idx, correct_prediction = eval_model_accuracy(
            NN=NN, train_ds=train_ds, test_ds=test_ds, holdout=holdout
        )
        
        holdout_x_list = [int(x * X_NORMALIZE_FACTOR) for x in holdout['x']]
        case_results = [case_num + 1] + holdout_x_list + \
            [holdout['y'], pred_idx, correct_prediction, acc_test]
        results.append(case_results)

    print_case_results(results=results, case_num=case_num, start_time=start_time)
    

if __name__ == '__main__':
    main()

