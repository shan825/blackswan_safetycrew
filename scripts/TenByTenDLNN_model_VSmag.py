# Author: Doug Cady
# Test Pytorch
# Python 3.10, Pytorch Using CPU
# Pytorch Version 1.12.1
# Based off example: https://visualstudiomagazine.com/articles/2022/09/06/multi-class-pytorch.aspx

import numpy as np
import random
import time
import pandas as pd
import torch as T
from sklearn.model_selection import train_test_split


INPUT_DATA_PATH = "../data/norm_tenByTenModelData.csv"
DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# Training/Testing Parameters
RANDOM_SEED = 111
TESTING_SPLIT_PERC = 0.3

# Model Parameters
MAX_EPOCHS = 16000
EP_LOG_INTERVAL = 500
BATCH_SIZE = 40
LEARNING_RATE = 0.03
NUM_FEATURES = 3
HIDDEN_LAYER_NODES = 64
NUM_CLASSES = 100

SAVED_MODEL_FN = f"../models/norm_10x10PyTorchNN3LEpochs{MAX_EPOCHS}Learn{LEARNING_RATE}.pt"


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
        self.hid1 = T.nn.Linear(NUM_FEATURES, 64)  # 3-(64-64)-100
        self.hid2 = T.nn.Linear(64, 64)
        self.oupt = T.nn.Linear(64, NUM_CLASSES)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        # Try tanh() or relu() to see which performs better
        z = T.tanh(self.hid1(x))
        z = T.tanh(self.hid2(z))
        z = T.log_softmax(self.oupt(z), dim=1)  # NLLLoss() 
        return z


def read_split_data():
    data = pd.read_csv(INPUT_DATA_PATH)
    print(f"[+] Read {len(data)} rows from csv file.")
    
    x_all = data.iloc[:, 0:NUM_FEATURES].to_numpy()
    y_all = data.iloc[:, NUM_FEATURES].to_numpy()
    return x_all, y_all


def split_train_test_holdout(x_all: np.array, y_all: np.array):
    """Split x, y into training, testing, and holdout datasets."""
    random.seed(RANDOM_SEED)
    holdout_row = random.randint(0, 799)
    holdout = {
        'x': x_all[holdout_row, :],
        'y': y_all[holdout_row],
    }

    x = np.delete(x_all, (holdout_row), axis=0)
    y = np.delete(y_all, (holdout_row))

    train = {}
    test = {}
    train['x'], test['x'], train['y'], test['y'] = train_test_split(x, y, test_size=TESTING_SPLIT_PERC, 
                                                              random_state=RANDOM_SEED)

    print(f"[+] Split into training, testing, and holdout datasets:")
    print(f"[+] - x Training: {len(train['x'])} rows ({len(train['x'])/len(x) * 100:.0f}%)")
    print(f"[+] - y Training: {len(train['y'])} rows ({len(train['y'])/len(y) * 100:.0f}%)")
    print(f"[+] - x Testing: {len(test['x'])} ({len(test['x'])/len(x) * 100:.0f}%)")
    print(f"[+] - y Testing: {len(test['y'])} ({len(test['y'])/len(y) * 100:.0f}%)")

    return train, test, holdout


def to_class_dataset(train, test):
    """Convert each dataset to a classifier dataset."""
    return (ModelDataset(train['x'], train['y']),
            ModelDataset(test['x'], test['y']))


# def calc_class_weights(y: pd.DataFrame):
#     """Calculate class weights based off target dataframe."""
#     print(f"[+] Calculating class weights...")
    
#     class_count = y.copy()
#     class_count['count'] = 1
#     class_df_gb = pd.DataFrame(class_count.groupby('target')['count'].count()).reset_index()
    
#     class_weights = 1. / T.tensor(class_df_gb['count'], dtype=T.float)
#     return class_weights


# def set_dataloader(class_weights_all, train_ds, test_ds, valid_ds):
#     """Set up data loaders from T tensors."""
#     weighted_sampler = WeightedRandomSampler(
#         weights=class_weights_all,
#         num_samples=len(class_weights_all),
#         replacement=True
#     )

#     return (DataLoader(dataset=train_ds,
#                        batch_size=BATCH_SIZE,
#                        sampler=weighted_sampler),
#             DataLoader(dataset=test_ds, batch_size=1),
#             DataLoader(dataset=valid_ds, batch_size=1))


def train_model(train_ds, test_ds):
    """Use input training data to train a neural network and testing data to calculate accuracy."""
    train_loader = T.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    print(f"[+] Creating {NUM_FEATURES}-({HIDDEN_LAYER_NODES}-{HIDDEN_LAYER_NODES})-{NUM_CLASSES} neural network...")
    NN = Net().to(DEVICE)
    NN.train()

    loss_func = T.nn.NLLLoss()  # assumes log_softmax()
    optimizer = T.optim.SGD(NN.parameters(), lr=LEARNING_RATE)

    print(f"[+] 3L, tanh, batch_size={BATCH_SIZE}, epochs={MAX_EPOCHS}, learn_rate={LEARNING_RATE}")
    print("[+] Training model...")

    for epoch in range(0, MAX_EPOCHS):
        T.manual_seed(epoch+1) # for testing purposes only
        epoch_loss = 0

        for batch in train_loader:
            X = batch[0] # inputs
            Y = batch[1] # target label

            optimizer.zero_grad()
            oupt = NN(X)
            loss_val = loss_func(oupt, Y) # a tensor
            epoch_loss += loss_val.item() # accumulate loss
            loss_val.backward()
            optimizer.step()

        if epoch % EP_LOG_INTERVAL == 0:
            print(f"[+] - epoch = {epoch:>5} | loss = {epoch_loss:8.4f}")

    print("[+] Training done")
    return NN


def calc_model_accuracy(model, ds):
    """Calculate prediction accuracy for a given model."""
    n_correct = 0
    n_wrong = 0

    for i in range(len(ds)):
        X = ds[i][0].reshape(1, -1) # make it a batch
        Y = ds[i][1].reshape(1) # 1 dim

        with T.no_grad():
            oupt = model(X) # logits form

        if T.argmax(oupt) == Y:
            n_correct += 1
        else:
            n_wrong += 1

    return (n_correct * 1.0) / (n_correct + n_wrong)


def eval_model_accuracy(NN, train_ds, test_ds, holdout, save_model):
    """Compute and evaluate accuracy of NN model on testing and holdout data."""
    NN.eval()
    acc_train = calc_model_accuracy(model=NN, ds=train_ds)
    print(f"[+] Training accuracy: {acc_train:0.4f}")

    acc_test = calc_model_accuracy(model=NN, ds=test_ds)
    print(f"[+] Testing accuracy: {acc_test:0.4f}")

    # Make prediction on holdout data
    print(f"[+] Predicting on holdout {holdout['x']} | Target index: {holdout['y']}:")
    torch_holdout = T.tensor([holdout['x']], dtype=T.float32).to(DEVICE)

    with T.no_grad():
        logits = NN(torch_holdout)
    probs = T.exp(logits).numpy()

    _, pred_idx = np.where(probs == np.amax(probs))
    print(f"[i] > Predicted target index: {pred_idx}")
    predict_res = "Correct" if holdout['y'] == pred_idx else "Incorrect"
    print(f"[i] > {predict_res} prediction")
    
    np.set_printoptions(precision=4, suppress=True)
    print(probs)

    if save_model:
        print("[+] Saving trained model state")
        T.save(NN.state_dict(), SAVED_MODEL_FN)


def main() -> int:
    start_time = time.time()
    x_all, y_all = read_split_data()
    train, test, holdout = split_train_test_holdout(x_all=x_all, y_all=y_all)
    train_ds, test_ds = to_class_dataset(train=train, test=test)

    TRAIN_NEW_MODEL = True

    if TRAIN_NEW_MODEL:
        NN = train_model(train_ds=train_ds, test_ds=test_ds)
    else:
        NN = Net()
        NN.load_state_dict(T.load(SAVED_MODEL_FN))

    eval_model_accuracy(NN=NN, train_ds=train_ds, test_ds=test_ds, holdout=holdout, save_model=TRAIN_NEW_MODEL)

    time_elapsed = time.time() - start_time
    minutes = time_elapsed // 60
    seconds = time_elapsed % 60
    print(f"[+] Program finished in {int(minutes)}m:{seconds:0.2f}s")


if __name__ == '__main__':
    main()