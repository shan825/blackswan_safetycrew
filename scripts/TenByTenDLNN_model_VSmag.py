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
import seaborn as sns
import matplotlib.pyplot as plt


INPUT_DATA_PATH = "../data/norm_tenByTenModelData.csv"
DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# Training/Testing Parameters
RANDOM_SEED = 24
TESTING_SPLIT_PERC = 0.3

# Model Parameters
ACTIVATION_FUNC = 'Relu'
HIDDEN_LAYER_NODE_1 = 512
HIDDEN_LAYER_NODE_2 = 128
HIDDEN_LAYER_NODE_3 = 64

MAX_EPOCHS = 2_000
EP_LOG_INTERVAL = MAX_EPOCHS / 4
BATCH_SIZE = 40
LEARNING_RATE = 0.037

NUM_FEATURES = 3
NUM_CLASSES = 100

SAVED_MODEL_FN = f"../models/PyTor_{ACTIVATION_FUNC}4L{HIDDEN_LAYER_NODE_1}-{HIDDEN_LAYER_NODE_2}-" + \
    f"{HIDDEN_LAYER_NODE_3}N{MAX_EPOCHS}E{LEARNING_RATE}Lr{BATCH_SIZE}Bs{RANDOM_SEED}Rs.pt"

TRAINING_LOSS_PLOTDATA_FN = f"../plot_data/PyTor_EpochPlot_4L{MAX_EPOCHS}E{LEARNING_RATE}Lr{RANDOM_SEED}Rs.csv"


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
        # self.dropout = T.nn.Dropout(p = 0.2)
        self.hid3 = T.nn.Linear(HIDDEN_LAYER_NODE_2, HIDDEN_LAYER_NODE_3)
        self.output = T.nn.Linear(HIDDEN_LAYER_NODE_3, NUM_CLASSES)

        # T.nn.init.xavier_uniform_(self.hid1.weight)
        # T.nn.init.zeros_(self.hid1.bias)
        # T.nn.init.xavier_uniform_(self.hid2.weight)
        # T.nn.init.zeros_(self.hid2.bias)
        # T.nn.init.xavier_uniform_(self.hid3.weight)
        # T.nn.init.zeros_(self.hid3.bias)
        # T.nn.init.xavier_uniform_(self.output.weight)
        # T.nn.init.zeros_(self.output.bias)

    def forward(self, x):
        # Try tanh() or relu() to see which performs better
        z = T.relu(self.hid1(x))
        z = T.relu(self.hid2(z))
        # z = self.dropout(z)
        z = T.relu(self.hid3(z))
        z = T.log_softmax(self.output(z), dim=1)  # NLLLoss() 
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
    train['x'], test['x'], train['y'], test['y'] = train_test_split(
        x, y, test_size=TESTING_SPLIT_PERC, random_state=RANDOM_SEED
    )

    print(f"[+] Split into training, testing, and holdout datasets")
    # print(f"[+] - x Training: {len(train['x'])} rows ({len(train['x'])/len(x) * 100:.0f}%)")
    # print(f"[+] - y Training: {len(train['y'])} rows ({len(train['y'])/len(y) * 100:.0f}%)")
    # print(f"[+] - x Testing: {len(test['x'])} ({len(test['x'])/len(x) * 100:.0f}%)")
    # print(f"[+] - y Testing: {len(test['y'])} ({len(test['y'])/len(y) * 100:.0f}%)")

    return train, test, holdout


def to_class_dataset(train, test):
    """Convert each dataset to a classifier dataset."""
    return (ModelDataset(train['x'], train['y']),
            ModelDataset(test['x'], test['y']))


def train_model(train_ds, test_ds):
    """Use input training data to train a neural network and testing data to calculate accuracy."""
    train_loader = T.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    print(f"[+] Creating {NUM_FEATURES}-({HIDDEN_LAYER_NODE_1}-{HIDDEN_LAYER_NODE_2}-{HIDDEN_LAYER_NODE_3})-{NUM_CLASSES} neural network...")
    NN = Net().to(DEVICE)
    NN.train()

    loss_func = T.nn.NLLLoss()  # assumes log_softmax()
    optimizer = T.optim.SGD(NN.parameters(), lr=LEARNING_RATE)
    loss_stats = []
    # scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    print(f"[+] 4L, {ACTIVATION_FUNC}, batch_size={BATCH_SIZE}, epochs={MAX_EPOCHS},", \
        f"learn_rate={LEARNING_RATE}, random_seed={RANDOM_SEED}")
    print("[+] Training model...")

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
            # scheduler.step(epoch_loss)  # change learning rate if training loss stagnates
            
        loss_stats.append(epoch_loss / len(train_loader))

        if epoch % EP_LOG_INTERVAL == 0:
            print(f"[+] - epoch = {epoch:>5} | loss = {epoch_loss:8.4f}")

    print("[+] Training done")
    return NN, loss_stats


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


def eval_model_accuracy(NN, train_ds, test_ds, holdout, save_model):
    """Compute and evaluate accuracy of NN model on testing and holdout data."""
    NN.eval()
    acc_train = calc_model_accuracy(model=NN, ds=train_ds)
    print(f"[+] Training accuracy: {acc_train * 100:0.2f}%")

    acc_test = calc_model_accuracy(model=NN, ds=test_ds)
    print(f"[+] Testing accuracy:  {acc_test * 100:0.2f}%")

    # Make prediction on holdout data
    print(f"[+] Predicting on holdout {holdout['x']} | Target index: {holdout['y']}:")
    torch_holdout = T.tensor(np.array([holdout['x']]), dtype=T.float32).to(DEVICE)

    with T.no_grad():
        logits = NN(torch_holdout)
    probs = T.exp(logits).numpy()

    _, pred_idx = np.where(probs == np.amax(probs))
    predict_res = "Correct" if holdout['y'] == pred_idx else "Incorrect"
    print(f"[i] > Predicted target index: {pred_idx} [ {predict_res} ]")
    
    np.set_printoptions(precision=4, suppress=True)
    print(probs)

    if save_model:
        print("[+] Saving trained model state")
        T.save(NN.state_dict(), SAVED_MODEL_FN)


def plot_loss(loss_stats) -> None:
    """Plot epoch loss stats over each epoch."""
    df_train_loss = (pd.DataFrame(loss_stats, columns=['loss'])
                     .reset_index()
                     .melt(id_vars=['index'])
                     .rename(columns={'index': 'epochs'}))
    # df_train_loss.to_csv(TRAINING_LOSS_PLOTDATA_FN, index=False)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 9))
    sns.lineplot(data=df_train_loss, x = "epochs", y = "value").set(title='Training Loss / Epoch')
    plt.show()


def print_time_elapsed(start_time: float) -> None:
    time_elapsed = time.time() - start_time
    
    if time_elapsed > 3600:
        hours = time_elapsed // 3600
        rest = time_elapsed % 3600
        minutes = rest // 60
        seconds = rest % 60
        print(f"[+] Program finished in {int(hours)}h, {int(minutes)}m, {seconds:0.2f}s")
    else:
        minutes = time_elapsed // 60
        seconds = time_elapsed % 60
        print(f"[+] Program finished in {int(minutes)}m, {seconds:0.2f}s")


def main() -> int:
    start_time = time.time()

    x_all, y_all = read_split_data()
    train, test, holdout = split_train_test_holdout(x_all=x_all, y_all=y_all)
    train_ds, test_ds = to_class_dataset(train=train, test=test)

    TRAIN_NEW_MODEL = True

    if TRAIN_NEW_MODEL:
        NN, loss_stats = train_model(train_ds=train_ds, test_ds=test_ds)
    else:
        NN = Net()
        NN.load_state_dict(T.load(SAVED_MODEL_FN))

    eval_model_accuracy(NN=NN, train_ds=train_ds, test_ds=test_ds, holdout=holdout, save_model=TRAIN_NEW_MODEL)

    print_time_elapsed(start_time=start_time)

    plot_loss(loss_stats=loss_stats)
    

if __name__ == '__main__':
    main()