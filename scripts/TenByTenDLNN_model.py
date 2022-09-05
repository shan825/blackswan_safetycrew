# Author Shing-Han
# Test Pytorch
# Python 3.10, Pytorch Using CUDA 10.6 Architecture
# Pytorch Version 1.12.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print(torch.__version__)

# Read in Data
df = pd.read_csv(r'C:\Users\Shing\Documents\pytorch_test\data\tenByTenModelData.csv')
print(df)
# Split into Train Test Validation and Test
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# print(X)
# print(y)
# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.4, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=21)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)

# print(target_list)
df['count'] = 1

class_dict_gb = pd.DataFrame(df.groupby('target')['count'].count()).reset_index()

class_count_dict = dict(zip(class_dict_gb['target'], class_dict_gb['count']))
# print(class_count_dict)
# Get class weights for target class
class_weights = 1. / torch.tensor(class_dict_gb['count'], dtype=torch.float)
print(class_weights)
# Get class weights for all 800 instances and assign them
# @Doug can you please explain what is going on in this below code I can't wrap my head around it
# I would have previously just did a join using the target # as an index
class_weights_all = class_weights[target_list]
print(class_weights_all)

# create weighted sampler
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

# Model Parameters
EPOCHS = 10000000
BATCH_SIZE = 2000
LEARNING_RATE = 0.0007
NUM_FEATURES = len(X.columns)
NUM_CLASSES = 100

# We instatiated the train_dataset earlier
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
                          )
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


# Create multiclassclassification class
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


model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)


# Accuracy Prediction Function
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


# Define 2 dictionaries that store accuracy/epoch and loss epoch for both train and validation
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

# Train Model
print("Begin training.")
for e in tqdm(range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
model.train()
for X_train_batch, y_train_batch in train_loader:
    X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
    optimizer.zero_grad()

    y_train_pred = model(X_train_batch)

    train_loss = criterion(y_train_pred, y_train_batch)
    train_acc = multi_acc(y_train_pred, y_train_batch)

    train_loss.backward()
    optimizer.step()

    train_epoch_loss += train_loss.item()
    train_epoch_acc += train_acc.item()

# VALIDATION
with torch.no_grad():
    val_epoch_loss = 0
    val_epoch_acc = 0

    model.eval()
    for X_val_batch, y_val_batch in val_loader:
        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

        y_val_pred = model(X_val_batch)

        val_loss = criterion(y_val_pred, y_val_batch)
        val_acc = multi_acc(y_val_pred, y_val_batch)

        val_epoch_loss += val_loss.item()
        val_epoch_acc += val_acc.item()
loss_stats['train'].append(train_epoch_loss / len(train_loader))
loss_stats['val'].append(val_epoch_loss / len(val_loader))
accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

print(
    f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')


# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear()
# x = torch.rand(64, 10)
# y = torch.rand(64, 10)
# # z = x+y
# print(z.shape)

# def main():
#     print("begin predicting end state")
#     np.random.seed(1)
#     torch.manual(seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = nn.Linear()
