import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

df = pd.read_csv('../dataset/demand.csv')

df = df.set_index('date')
df['OT'] = df['OT'].astype(float)

arima_model = sm.tsa.ARIMA(df.OT, order=(6, 1, 0)).fit()

print(arima_model.summary())

df['forecast'] = arima_model.predict(start=len(df) - 30, end=len(df), dynamic=True)

preds = np.array(df["forecast"][len(df) - 30:len(df)])
trues = np.array(df["OT"][len(df) - 30:len(df)])

mse = np.mean((preds - trues) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(preds - trues))
rse = np.sqrt(np.sum((trues - preds) ** 2)) / np.sqrt(np.sum((trues - trues.mean()) ** 2))

print('rmse: {}, mse:{}, mae:{}, rse:{}'.format(rmse, mse, mae, rse))

plt.figure(figsize=(12, 6))
plt.plot(df["OT"][len(df) - 30:len(df)], label='True Value')
plt.plot(df["forecast"][len(df) - 30:len(df)], label='Prediction Value')
plt.xticks(df["OT"][len(df) - 30:len(df)].index, list(range(1, 31)))
plt.title('Comparison between Prediction and True Value')
plt.legend()
plt.show()
