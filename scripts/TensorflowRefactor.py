# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:28:40 2022

@author: reiva
"""

import numpy as np
import random
import time
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
import xlsxwriter
from datetime import datetime


TODAYS_DATETIME = datetime.now().strftime("%m-%d-%Y_%H%M%S")
TODAYS_DATE = datetime.now().strftime("%m-%d-%Y")
TODAYS_TIME = datetime.now().strftime("%H:%M:%S %Z")

USER = 'Kelly'
INPUT_DATA_PATH = "../data/bounce_tenByTenModelData.csv"
# If you have tensorflow-gpu installed, then using the GPU 
# is enabled and done by default in Keras
DEVICE = tf.device('/cpu:0')
# DEVICE = tf.device("cuda:0" if tf.cuda.is_available() else "cpu")

TESTING_SPLIT_PERC = 0.25

# Model Parameters
KERNEL_INITIALIZER = 'he_uniform'
OPTIMIZER_STRING = 'SGD'
ACTIVATION_FUNC = 'relu'
OUT_ACTIVATION_FUNC = 'softmax'
HIDDEN_LAYER_NODE_1 = 512
HIDDEN_LAYER_NODE_2 = 256
HIDDEN_LAYER_NODE_3 = 128

MAX_EPOCHS = 80
EP_LOG_INTERVAL = MAX_EPOCHS / 4
BATCH_SIZE = 40
LEARNING_RATE = 0.02

X_NORMALIZE_FACTOR = 10
NUM_FEATURES = 3
NUM_CLASSES = 100
NUM_CASES = 800

# Accuracy and Loss metric for Training/Testing
# FILE_WRITER = tf.summary.create_file_writer("../logs/")   
TRAIN_ACC = []
TRAIN_LOSS = []
VAL_ACC = []
VAL_LOSS = []

LOSS_FUNC = tf.keras.losses.SparseCategoricalCrossentropy()
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

CASE_RESULTS_FN = f"TensorFlow_800caseResultsKelly__{TODAYS_DATETIME}.xlsx"


class ModelDataset(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data, batch_size = BATCH_SIZE):#, shuffle=True):
        self.batch_size = batch_size
        self.x_data = x_data
        self.y_data = y_data
        self.datalen = len(y_data)
        self.indexes = np.arange(self.datalen)
        # self.shuffle:
            #np.random.shuffle(self.indexes)

    def __len__(self):
        return self.datalen // self.batch_size
    
    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        predictions = self.x_data[batch_indexes]
        targets = self.y_data[batch_indexes]
        return predictions, targets

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)


def read_split_data():
    data = pd.read_csv(INPUT_DATA_PATH)
    print(f"[+] Read {len(data)} rows from csv file.")
    
    x_all = data.iloc[:, 0:NUM_FEATURES].to_numpy()
    y_all = data.iloc[:, NUM_FEATURES].to_numpy()
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
        x, y, test_size=TESTING_SPLIT_PERC#, random_state=RANDOM_SEED
    )

    return train, test, holdout


def to_class_dataset(train,test):
    """Convert each dataset to a classifier."""
    return (ModelDataset(train['x'], train['y']),
            ModelDataset(test['x'], test['y']))


@tf.function
def train_step(x, y, NN):
    with tf.GradientTape() as tape:
        logits = NN(x, training=True)
        loss_value = LOSS_FUNC(y, logits)
    grads = tape.gradient(loss_value, NN.trainable_weights)
    OPTIMIZER.apply_gradients(zip(grads, NN.trainable_weights))
    # acc_value = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    # print("Y")
    # print(y)

    # print("logits")
    # print(tf.argmax(logits, 1))

    # import pdb; pdb.set_trace()

    # acc_value = tf.math.equal(y, tf.argmax(logits, 1))
    # acc_value = tf.math.equal(y, tf.cast(tf.math.round(tf.keras.activations.sigmoid(logits)), tf.int64))
    # TRAIN_ACC.append(acc_value)
    TRAIN_LOSS.append(loss_value)

@tf.function
def test_step(x, y, NN):
    val_logits = NN(x, training=False)
    loss_value = LOSS_FUNC(y, val_logits)
    # acc_value = tf.math.equal(y, tf.argmax(val_logits, 1))
    # acc_value = tf.math.equal(y, tf.cast(tf.math.round(tf.keras.activations.sigmoid(val_logits)), tf.int64))
    # VAL_ACC.append(acc_value)
    VAL_LOSS.append(loss_value)
    # return loss_value, acc_value


def train_model(train_ds, test):
    """Use input training data to train a neural network."""
    NN = Sequential([
        Dense(HIDDEN_LAYER_NODE_1, input_dim=NUM_FEATURES, activation=ACTIVATION_FUNC, 
            kernel_initializer=KERNEL_INITIALIZER),
        Dense(HIDDEN_LAYER_NODE_2, activation=ACTIVATION_FUNC, kernel_initializer=KERNEL_INITIALIZER),
        Dense(HIDDEN_LAYER_NODE_3, activation=ACTIVATION_FUNC, kernel_initializer=KERNEL_INITIALIZER),
        Dense(NUM_CLASSES, activation=OUT_ACTIVATION_FUNC, kernel_initializer=KERNEL_INITIALIZER)
    ])

    # mODEL.fit way:

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_ds))
    # train_loader = (train_dataset.shuffle(len(train_ds))
    # .batch(BATCH_SIZE))
    
    # NN.compile(
    #     loss="sparse_categorical_crossentropy",
    #     optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE),
    #     metrics=["acc"],).to(DEVICE)

    # history = NN.fit(
    #     trains_ds,
    #     validation_data=test_ds,
    #     validation_steps=len(test_ds[0]) // BATCH_SIZE,
    #     epochs=MAX_EPOCHS
    # )


    # Explicit manual loop way:
    for epoch in range(0, MAX_EPOCHS):
        for inputs, targets in train_ds:
            train_step(x=inputs, y=targets, NN=NN)

        # print(test)
        # import pdb; pdb.set_trace()

        for batch_idx in range(0, len(test['y']), BATCH_SIZE):
            inputs = test['x'][batch_idx:batch_idx + BATCH_SIZE]
            targets = test['y'][batch_idx:batch_idx + BATCH_SIZE]

            # print(test)
            # print(inputs)
            # print(targets)
            # import pdb; pdb.set_trace()

            test_step(x=inputs, y=targets, NN=NN)

    # with FILE_WRITER.as_default():
    #     tf.summary.scalar('Loss/train', TRAIN_LOSS.result(), step=epoch)
    #     tf.summary.scalar('Loss/validation', VAL_LOSS.result(), step=epoch)
    #     tf.summary.scalar('Accuracy/train', TRAIN_ACC.result(), step=epoch)
    #     tf.summary.scalar('Accuracy/validation', VAL_ACC.result(), step=epoch)

    # TRAIN_ACC.reset_states()
    # TRAIN_LOSS.reset_states()
    # VAL_LOSS.reset_states()
    # VAL_ACC.reset_states()

    return NN
    

def calc_model_accuracy(NN, ds):
    """Calculate prediction accuracy for a given NN."""
    n_correct = 0
    n_wrong = 0

    for i in range(len(ds)):
        # X = ds[i]
        # Y = ds[i][1].reshape(1) # 1 dim

        # with tf.GradientTape() as tape:
        #     logits = NN(ds, training=True)
        # output = NN(X) # logits form

        print(ds[i][0])
        print(logits)
        import pdb; pdb.set_trace()

        if tf.argmax(logits) == Y:
            n_correct += 1
        else:
            n_wrong += 1

    return (n_correct * 1.0) / (n_correct + n_wrong)


def eval_model_accuracy(NN, train_ds, test_ds, holdout):
    """Compute and evaluate accuracy of NN NN on testing and holdout data."""
    acc_train = calc_model_accuracy(NN=NN, ds=train_ds)
    acc_test = calc_model_accuracy(NN=NN, ds=test_ds)

    # Make prediction on holdout data
    tf_holdout = tf.convert_to_tensor(np.array([holdout['x']]), dtype=tf.float32).to(DEVICE)

    with T.no_grad():
        logits = NN(tf_holdout)
    probs = np.exp(logits)

    _, pred_idx = np.where(probs == np.amax(probs))
    correct_pred_idx = holdout['y'] == pred_idx[0]
    predict_res = "Correct" if correct_pred_idx else "*Incorrect*"

    # results = f"[i] > Training accuracy: {acc_train * 100:0.2f}% | " + \
    #     f"Testing: {acc_test * 100:0.2f}% | " + \
    #     f"Prediction: {pred_idx[0]:>2} [ {predict_res} ]"
    # print(results)

    return acc_test, pred_idx[0], correct_pred_idx

# =============================================================================
# def eval_model_accuracy(NN, train_ds, holdout):
#     """Compute and evaluate accuracy of NN model on holdout data."""
#     NN.eval()
#     acc_train = calc_model_accuracy(model=NN, ds=train_ds)
#     acc_test = calc_model_accuracy(model=NN, ds=test_ds)
# 
#     # Make prediction on holdout data
#     torch_holdout = tf.tensor(np.array([holdout['x']]), dtype=tf.float32).to(DEVICE)
# 
#     with tf.no_grad():
#         logits = NN(torch_holdout)
#     probs = tf.exp(logits).cpu().numpy()
# 
#     _, pred_idx = np.where(probs == np.amax(probs))
#     correct_pred_idx = holdout['y'] == pred_idx[0]
#     predict_res = "Correct" if correct_pred_idx else "*Incorrect*"
# 
#     return pred_idx[0], correct_pred_idx
# =============================================================================


def write_model_params(xl_writer):
    """Write model parameters to Excel file."""
    model_params = [
        ['User', USER],
        ['Date', TODAYS_DATE],
        ['Time', TODAYS_TIME],
        ['RunTime', 1],
        ['Packages', 'tensorflow'],
        ['Dataset', '10x10'],
        ['ActivationFunction', ACTIVATION_FUNC],
        ['Optimizer', OPTIMIZER_STRING],
        #['RandomSeed', RANDOM_SEED],
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

    # for case_num in range(NUM_CASES):
    for case_num in range(1):
        train, test, holdout = split_train_test_holdout(x_all=x_all, y_all=y_all, case_num=case_num)
        train_ds, test_ds = to_class_dataset(train=train, test=test)
        
        if case_num % 25 == 0:
            print(f"\n[+] Case # {case_num + 1:>3}: holdout {holdout['x'] * 10} | Target: {holdout['y']}")

        NN = train_model(train_ds=train_ds, test=test)

        acc_test, pred_idx, correct_prediction = eval_model_accuracy(
            NN=NN, train_ds=train_ds, test_ds=test_ds, holdout=holdout
        )
        
        # holdout_x_list = [int(x * X_NORMALIZE_FACTOR) for x in holdout['x']]
        # case_results = [case_num + 1] + holdout_x_list + \
        #     [holdout['y'], pred_idx, correct_prediction, acc_test]
        # results.append(case_results)

    # print_case_results(results=results, case_num=case_num, start_time=start_time)
    

if __name__ == '__main__':
    main()
