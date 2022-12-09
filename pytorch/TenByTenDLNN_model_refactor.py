# Author Shing-Han
# Test Pytorch
# Python 3.10, Pytorch Using CUDA 10.6 Architecture
# Pytorch Version 1.12.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


INPUT_DATA_PATH = "../data/tenByTenModelData.csv"
INPUT_FEATURES = ['xTemp', 'yVolume', 'direction']
TARGET_VARIABLE = ['target']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model Parameters
EPOCHS = 10000000
BATCH_SIZE = 2000
LEARNING_RATE = 0.0007
NUM_CLASSES = 100
NUM_FEATURES = len(INPUT_FEATURES)


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


def read_split_data():
    data = pd.read_csv(INPUT_DATA_PATH)
    print(f"[+] Read {len(data)} rows from csv file.")
    return data[INPUT_FEATURES], data[TARGET_VARIABLE]


def to_np_array(X: pd.DataFrame, y: pd.DataFrame):
    """Convert input dataframes to numpy arrays."""
    return np.array(X), np.array(y)


def split_train_test_validation(X: pd.DataFrame, y: pd.DataFrame):
    """Split X, y into training, testing, and validation datasets."""
    train = {}
    test = {}
    validation = {}

    X_trainval, test['X'], y_trainval, test['y'] = train_test_split(
        X, y, test_size=0.4, random_state=69)
    train['X'], validation['X'], train['y'], validation['y'] = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=21
    )

    # Scale input features
    scaler = MinMaxScaler()
    train['X'] = scaler.fit_transform(train['X'])
    test['X'] = scaler.transform(test['X'])
    validation['X'] = scaler.transform(validation['X'])

    # Convert to all variables to numpy arrays for future modeling step
    for data_dict in [train, test, validation]:
        data_dict['X'], data_dict['y'] = to_np_array(X=data_dict['X'], y=data_dict['y'])

    print(f"[+] Split into training, testing, and validation datasets:")
    print(f"[+] - Training: {len(train['X'])} rows ({len(train['X'])/len(X) * 100:.0f}%)")
    print(f"[+] - Testing: {len(test['X'])} ({len(test['X'])/len(X) * 100:.0f}%)")
    print(f"[+] - Validation: {len(validation['X'])} ({len(validation['X'])/len(X) * 100:.0f}%)")

    return train, test, validation


def torch_class_dataset(train, test, validation):
    """Convert each dataset to a classifier dataset."""
    return (ClassifierDataset(torch.from_numpy(train['X']).float(), 
                              torch.from_numpy(train['y']).long()),

            ClassifierDataset(torch.from_numpy(test['X']).float(), 
                              torch.from_numpy(test['y']).long()),
    
            ClassifierDataset(torch.from_numpy(validation['X']).float(), 
                              torch.from_numpy(validation['y']).long()))


def calc_class_weights(y: pd.DataFrame):
    """Calculate class weights based off target dataframe."""
    print(f"[+] Calculating class weights...")
    
    class_count = y.copy()
    class_count['count'] = 1
    class_df_gb = pd.DataFrame(class_count.groupby('target')['count'].count()).reset_index()
    
    class_weights = 1. / torch.tensor(class_df_gb['count'], dtype=torch.float)
    return class_weights


def set_dataloader(class_weights_all, torch_train, torch_test, torch_valid):
    """Set up data loaders from torch tensors."""
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    return (DataLoader(dataset=torch_train,
                       batch_size=BATCH_SIZE,
                       sampler=weighted_sampler),
            DataLoader(dataset=torch_test, batch_size=1),
            DataLoader(dataset=torch_valid, batch_size=1))


def calc_pred_accuracy(y_pred, y_test):
    """Calculate prediction accuracy for a given model."""
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return torch.round(acc * 100)


def train_model(model, class_weights, train_loader, valid_loader):
    print(f"[+] Training NN model...")
    loss_stats = {
        'train': [],
        'val': []
    }
    accuracy_stats = {
        'train': [],
        'val': []
    }

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # print(model)

    # @Shing, I thought the indent might be messed up on the example we're working from
    # so I am guessing it looks more like this below, but this hasn't worked for me.
    for e in range(1, EPOCHS + 1):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()

        for X_train_batch, y_train_batch in train_loader:
            X_train_batch_dev = X_train_batch.to(DEVICE)
            y_train_batch_dev = y_train_batch.to(DEVICE)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch_dev)
            train_loss = criterion(y_train_pred, y_train_batch_dev)
            train_acc = calc_pred_accuracy(y_pred=y_train_pred, y_test=y_train_batch_dev)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # Validation
        with torch.no_grad():
            valid_epoch_loss = 0
            valid_epoch_acc = 0
            model.eval()

            for X_valid_batch, y_valid_batch in valid_loader:
                X_valid_batch_dev = X_valid_batch.to(DEVICE)
                y_valid_batch_dev = y_valid_batch.to(DEVICE)

                y_valid_pred = model(X_valid_batch_dev)
                valid_loss = criterion(y_valid_pred, y_valid_batch_dev)
                valid_acc = calc_pred_accuracy(y_pred=y_valid_pred, y_test=y_valid_batch_dev)

                valid_epoch_loss += valid_loss.item()
                valid_epoch_acc += valid_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        loss_str = f"TrainLoss: {train_epoch_loss / len(train_loader):.5f} | ValidLoss: {valid_epoch_loss / len(valid_loader):.5f}"
        accuracy_str = f"TrainAcc: {train_epoch_acc / len(train_loader):.5f} | ValidAcc: {valid_epoch_acc / len(valid_loader):.5f}"
        print(f"e: {e + 0:03} | {loss_str} | {accuracy_str}")


def main() -> int:
    X, y = read_split_data()
    train, test, validation = split_train_test_validation(X=X, y=y)
    torch_train, torch_test, torch_valid = torch_class_dataset(train=train, test=test, 
                                                               validation=validation)

    target_list = torch.tensor([yTarget for _, yTarget in torch_train])
    class_weights = calc_class_weights(y=y)
    # print("Num class weights: ", len(class_weights))
    class_weights_all = class_weights[target_list]
    
    train_loader, test_loader, valid_loader = set_dataloader(class_weights_all=class_weights_all,
                                                             torch_train=torch_train,
                                                             torch_test=torch_test,
                                                             torch_valid=torch_valid)

    model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(DEVICE)

    train_model(model=model, class_weights=class_weights, train_loader=train_loader, valid_loader=valid_loader)


if __name__ == '__main__':
    main()