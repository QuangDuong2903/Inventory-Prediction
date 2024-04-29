import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from lstm import LSTMModel

sequence_length = 90
prediction_length = 30
batch_size = 128
epochs = 100
device = 'cpu'
learning_rate = 0.001


def to_sequences(data):
    x = []
    y = []
    for i in range(len(data) - sequence_length - prediction_length):
        window = data[i:(i + sequence_length)]
        after_window = data[i + sequence_length: i + sequence_length + prediction_length]
        x.append(window)
        yy = []
        for rec in after_window:
            yy.append(rec[len(rec) - 1])
        y.append(yy)
    x = np.array(x)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def vali(loader):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)

            loss = criterion(outputs, y_batch)

            total_loss.append(loss)

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


df = pd.read_csv('../dataset/demand.csv')

df.drop('date', axis=1, inplace=True)

num_train = int(len(df) * 0.7)
num_test = int(len(df) * 0.2)
num_vali = len(df) - num_train - num_test

df_train = df[0:num_train].to_numpy()
df_vali = df[num_train - sequence_length:num_train + num_vali].to_numpy()
df_test = df[len(df) - num_test - sequence_length: len(df)].to_numpy()

scaler = StandardScaler()

df_train = scaler.fit_transform(df_train)
df_vali = scaler.fit_transform(df_vali)
df_test = scaler.fit_transform(df_test)

x_train, y_train = to_sequences(df_train)
x_val, y_val = to_sequences(df_vali)
x_test, y_test = to_sequences(df_test)

# Define dataset
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

# Define data loader
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

model = LSTMModel().to(device)

# Train the model
train_steps = len(train_loader)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

early_stop_count = 0
min_val_loss = float('inf')

for epoch in range(epochs):
    train_loss = []
    model.train()
    epoch_time = time.time()
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_loss = np.average(train_loss)
    vali_loss = vali(val_loader)
    test_loss = vali(test_loader)

    scheduler.step(vali_loss)

    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        epoch + 1, train_steps, train_loss, vali_loss, test_loss))

    if vali_loss < min_val_loss:
        min_val_loss = vali_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 10:
        print("Early stopping!")
        break

preds = []
trues = []
inputx = []

folder_path = './results/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

model.eval()
with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.float().to(device), y_batch.float().to(device)

        outputs = model(x_batch)

        outputs = outputs.detach().cpu().numpy()
        y_batch = y_batch.detach().cpu().numpy()

        pred = outputs
        true = y_batch

        preds.extend(outputs.squeeze().tolist())
        trues.extend(y_batch.squeeze().tolist())
        inputx.extend(x_batch.squeeze().tolist())

preds = np.array(preds)
trues = np.array(trues)
inputx = np.array(inputx)

np.save(folder_path + 'pred.npy', preds)
np.save(folder_path + 'true.npy', trues)
np.save(folder_path + 'x.npy', inputx)

mse = np.mean((preds - trues) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(preds - trues))
rse = np.sqrt(np.sum((trues - preds) ** 2)) / np.sqrt(np.sum((trues - trues.mean()) ** 2))

print('rmse: {}, mse:{}, mae:{}, rse:{}'.format(rmse, mse, mae, rse))

torch.save(model, folder_path + 'model.pth')

f = open("result.txt", 'a')
f.write('rmse: {}, mse:{}, mae:{}, rse:{}'.format(rmse, mse, mae, rse))
f.write('\n')
f.write('\n')
f.close()

trues = np.load('./results/true.npy')

preds = np.load('./results/pred.npy')

inputx = np.load('./results/x.npy')

for i, x in enumerate(inputx):
    true = []
    for d in x:
        true.append(d[13])
    true.extend(trues[i])
    plt.plot(true, label='True Value')
    pred = preds[i]
    plt.plot(range(sequence_length, sequence_length + prediction_length), preds[i], label='Prediction Value')
    plt.title('Comparison between Prediction and True Value')
    plt.legend()
    plt.savefig(folder_path + 'pred-{}.png'.format(i + 1))
    plt.clf()
