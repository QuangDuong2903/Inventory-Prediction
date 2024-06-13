import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('dataset/Store Item Demand.csv')

df['date'] = pd.to_datetime(df['date'])

df = df[df['item'] == 1]

df['OT'] = df.groupby('date')['sales'].transform('sum')

df.drop('store', axis=1, inplace=True)
df.drop('item', axis=1, inplace=True)
df.drop('sales', axis=1, inplace=True)
df = df.drop_duplicates(subset=['date'])

# datetime features
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['weekday'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week
df['weekend'] = df.apply(lambda x: 1 if x['weekday'] > 4 else 0, axis=1)
df['holidays'] = df.apply(lambda x: 1 if holidays.country_holidays('VN').get(x['date']) else 0, axis=1)
df['quarter'] = df['date'].dt.quarter
# Cyclical Features
df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))

# historical / seasonal features
df["median-month"] = df.groupby('month')['OT'].transform('median')
df["mean-week"] = df.groupby('week_of_year')['OT'].transform('mean')
# total sales of that item
df["item-month-sum"] = df.groupby('month')['OT'].transform('sum')

# shifted features

# sales for that item 90 days = 3 months ago
# df['OT-shifted-90'] = df['OT'].shift(90).fillna(df['OT'].interpolate())
# sales for that item 180 days = 6 months ago
# df['OT-shifted-180'] = df['OT'].shift(180).fillna(df['OT'].interpolate())
# sales for that item 1 year ago
# df['OT-shifted-365'] = df['OT'].shift(365).fillna(df['OT'].interpolate())

df['date'] = df['date'].dt.strftime('%-m/%-d/%Y %-H:%M')

ot = df.pop('OT')
df['OT'] = ot

df.to_csv('demand.csv', index=False)

df = pd.read_csv('dataset/demand.csv')

df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['OT'], color='blue')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales over Time')
plt.grid(True)
plt.show()

df = df.set_index('date')
y = df['OT'].resample('MS').mean()
y.plot(figsize=(12, 5))
plt.show()

# # Decomposition
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

# # AutoCorrelation
plt.figure(figsize=(8, 6))
ax1 = plot_acf(y)
plt.show()

# # PartialCorrelation
plt.figure(figsize=(8, 6))
ax2 = plot_pacf(y)
plt.show()
