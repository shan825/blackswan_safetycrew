"""
10x10 Neural Network Multi-class Classification
Loop over 800 input cases:
- Take 1 holdout case
- Train on 70%, test on 30%
- Print testing accuracy and holdout accuracy at end

Author: Doug Cady
Python 3.10, Pytorch Using CPU
Pytorch Version 1.12.1
Based off example: https://visualstudiomagazine.com/articles/2022/09/06/multi-class-pytorch.aspx

"""


import numpy as np
import random
import time
import pandas as pd
import torch as T
from sklearn.model_selection import train_test_split
import xlsxwriter
from datetime import datetime

USER = 'Doug'

TODAYS_DATETIME = datetime.now().strftime("%m-%d-%Y_%H%M%S")
TODAYS_DATE = datetime.now().strftime("%m-%d-%Y")
TODAYS_TIME = datetime.now().strftime("%H:%M:%S %Z")

INPUT_DATA_PATH = "../data/norm_tenByTenModelData.csv"
# DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")
DEVICE = "cpu"

# Training/Testing Parameters
RANDOM_SEED = (datetime.now() - datetime(2022, 8, 1)).total_seconds()
TESTING_SPLIT_PERC = 0.2

# Model Parameters
ACTIVATION_FUNC = 'Relu'
HIDDEN_LAYER_NODE_1 = 512
HIDDEN_LAYER_NODE_2 = 256
HIDDEN_LAYER_NODE_3 = 128
HIDDEN_LAYER_NODE_4 = 64

MAX_EPOCHS = 2_500
EP_LOG_INTERVAL = MAX_EPOCHS / 4
BATCH_SIZE = 40
LEARNING_RATE = 0.2
OPTIMIZER = 'SGD'
MOMENTUM = 0.9

X_NORMALIZE_FACTOR = 10
NUM_FEATURES = 3
NUM_CLASSES = 100
NUM_CASES = 800

CASE_RESULTS_FN = f"../model_output/PyTorch_800caseResults__{TODAYS_DATETIME}.xlsx"


class ModelDataset(T.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = T.tensor(x_data, dtype=T.float32).to(DEVICE)
        self.y_data = T.tensor(y_data, dtype=T.int64).to(DEVICE)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        predictions = self.x_data[idx]
        targets = self.y_data[idx]
        return predictions, targets


class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(NUM_FEATURES, HIDDEN_LAYER_NODE_1)   # 3-(512-128-64)-100
        self.hid2 = T.nn.Linear(HIDDEN_LAYER_NODE_1, HIDDEN_LAYER_NODE_2)
        self.hid3 = T.nn.Linear(HIDDEN_LAYER_NODE_2, HIDDEN_LAYER_NODE_3)
        self.hid4 = T.nn.Linear(HIDDEN_LAYER_NODE_3, HIDDEN_LAYER_NODE_4)
        self.output = T.nn.Linear(HIDDEN_LAYER_NODE_4, NUM_CLASSES)


    def forward(self, x):
        # Try tanh() or relu() to see which performs better
        z = T.relu(self.hid1(x))
        z = T.relu(self.hid2(z))
        z = T.relu(self.hid3(z))
        z = T.relu(self.hid4(z))
        z = T.log_softmax(self.output(z), dim=1)  # NLLLoss() 
        return z


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

    random.seed(RANDOM_SEED)
    rand_num = random.randint(0, 1_000_000_000)
    train['x'], test['x'], train['y'], test['y'] = train_test_split(
        x, y, test_size=TESTING_SPLIT_PERC, random_state=rand_num
    )

    return train, test, holdout


def to_class_dataset(train, test):
    """Convert each dataset to a classifier PyTorch Dataset."""
    return (ModelDataset(train['x'], train['y']),
            ModelDataset(test['x'], test['y']))


def train_model(train_ds, test_ds):
    """Use input training data to train a neural network and testing data to calculate accuracy."""
    train_loader = T.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    NN = Net().to(DEVICE)
    NN.train()

    loss_func = T.nn.NLLLoss()  # assumes log_softmax()
    optimizer = T.optim.SGD(NN.parameters(), lr=LEARNING_RATE)

    for epoch in range(0, MAX_EPOCHS):
        T.manual_seed(epoch + 1) # for testing purposes only
        epoch_loss = 0

        for batch in train_loader:
            X = batch[0] # inputs
            Y = batch[1] # target label

            optimizer.zero_grad()
            output = NN(X)
            loss_val = loss_func(output, Y) # a tensor
            epoch_loss += loss_val.item() # accumulate loss
            loss_val.backward()
            optimizer.step()
            
    return NN


def calc_model_accuracy(model, ds):
    """Calculate prediction accuracy for a given model."""
    n_correct = 0
    n_wrong = 0

    for i in range(len(ds)):
        X = ds[i][0].reshape(1, -1) # make it a batch
        Y = ds[i][1].reshape(1) # 1 dim

        with T.no_grad():
            output = model(X) # logits form

        if T.argmax(output) == Y:
            n_correct += 1
        else:
            n_wrong += 1

    return (n_correct * 1.0) / (n_correct + n_wrong)


def eval_model_accuracy(NN, train_ds, test_ds, holdout):
    """Compute and evaluate accuracy of NN model on testing and holdout data."""
    NN.eval()
    acc_train = calc_model_accuracy(model=NN, ds=train_ds)
    acc_test = calc_model_accuracy(model=NN, ds=test_ds)

    # Make prediction on holdout data
    torch_holdout = T.tensor(np.array([holdout['x']]), dtype=T.float32).to(DEVICE)

    with T.no_grad():
        logits = NN(torch_holdout)
    probs = T.exp(logits).numpy()

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
        ['Packages', 'PyTorch'],
        ['Dataset', '10x10'],
        ['ActivationFunction', ACTIVATION_FUNC],
        ['Optimizer', OPTIMIZER],
        ['Momentum', MOMENTUM],
        ['RandomSeed', 'time'],
        ['BatchSize', BATCH_SIZE],
        ['Epochs', MAX_EPOCHS],
        ['InitLayerWeightBias', 'No'],
        ['Layers', 4],
        ['NodesPerLayer', f"{HIDDEN_LAYER_NODE_1}, {HIDDEN_LAYER_NODE_2}, {HIDDEN_LAYER_NODE_3}, {HIDDEN_LAYER_NODE_4}"],
        ['NetworkArch', f"{NUM_FEATURES}-({HIDDEN_LAYER_NODE_1}-{HIDDEN_LAYER_NODE_2}-{HIDDEN_LAYER_NODE_3}-{HIDDEN_LAYER_NODE_4})-{NUM_CLASSES}"],
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
        train_ds, test_ds = to_class_dataset(train=train, test=test)
        
        if case_num % 25 == 0:
            print(f"\n[+] Case # {case_num + 1:>3}: holdout {holdout['x'] * 10} | Target: {holdout['y']}")

        NN = train_model(train_ds=train_ds, test_ds=test_ds)
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
