from _datetime import datetime, timedelta
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

from linear_regression import LinearRegressionModel

sequence_length = 90
prediction_length = 30
batch_size = 128
epochs = 300
device = 'cpu'


def to_sequences(data, scaler_x, scaler_y):
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
    x = x.reshape(x.shape[0], -1)
    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)
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

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train, y_train = to_sequences(df_train, scaler_x, scaler_y)
x_val, y_val = to_sequences(df_vali, scaler_x, scaler_y)
x_test, y_test = to_sequences(df_test, scaler_x, scaler_y)

# Define dataset
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

# Define data loader
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

model = LinearRegressionModel().to(device)

train_steps = len(train_loader)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                    steps_per_epoch=train_steps,
                                    pct_start=0.3,
                                    epochs=epochs,
                                    max_lr=0.0001)

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

    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        epoch + 1, train_steps, train_loss, vali_loss, test_loss))

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
trues = np.round(scaler_y.inverse_transform(trues)).astype(np.int32)

preds = np.load('./results/pred.npy')
preds = np.round(scaler_y.inverse_transform(preds)).astype(np.int32)

inputx = np.load('./results/x.npy')
inputx = scaler_x.inverse_transform(inputx)
inputx = np.round(inputx.reshape(335, 90, 14)).astype(np.int32)

for i, seq in enumerate(inputx):
    dates = []
    y = []
    for p in seq:
        dates.append(datetime.strptime('{}-{}-{}'.format(p[2], p[1], p[0]), "%Y-%m-%d"))
        y.append(p[13])
    y = np.append(y, trues[i])
    for _ in range(prediction_length):
        last_date = dates[-1]
        dates.append(last_date + timedelta(days=1))
    plt.plot(dates, y, label='True Value')
    pred = preds[i]
    plt.plot(dates[90:], pred, label='Prediction Value')
    plt.xlabel('Date')
    plt.xticks(rotation=20)
    plt.ylabel('Value')
    plt.title('Comparison between Prediction and True Value')
    plt.legend()
    plt.savefig(folder_path + 'pred-{}.png'.format(i + 1))
    plt.clf()
