import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
import urllib

import datetime

def round_to_hour(dt):
    dt_start_of_hour = dt.replace(minute=0, second=0, microsecond=0)
    dt_half_hour = dt.replace(minute=30, second=0, microsecond=0)

    if dt >= dt_half_hour:
        # round up
        dt = dt_start_of_hour + datetime.timedelta(hours=1)
    else:
        # round down
        dt = dt_start_of_hour

    return dt

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

data = pd.read_csv("combined_energy_data.csv", low_memory=False)
data = data.drop_duplicates(subset=['datetime'])
data['datetime'] = data['datetime'].apply(lambda x: x.split("+")[0])
data['datetime'] = pd.to_datetime(data['datetime'])

columns_use = ['datetime','Wind', 'guitrancourt_Speed(m/s)', 'guitrancourt_Direction (deg N)',
       'lieusaint_Speed(m/s)', 'lieusaint_Direction (deg N)',
       'lvs-pussay_Speed(m/s)', 'lvs-pussay_Direction (deg N)',
       'parc-du-gatinais_Speed(m/s)', 'parc-du-gatinais_Direction (deg N)',
       'arville_Speed(m/s)', 'arville_Direction (deg N)',
       'boissy-la-riviere_Speed(m/s)', 'boissy-la-riviere_Direction (deg N)',
       'angerville-1_Speed(m/s)', 'angerville-1_Direction (deg N)',
       'angerville-2_Speed(m/s)', 'angerville-2_Direction (deg N)',
       'guitrancourt-b_Speed(m/s)', 'guitrancourt-b_Direction (deg N)',
       'lieusaint-b_Speed(m/s)', 'lieusaint-b_Direction (deg N)',
       'lvs-pussay-b_Speed(m/s)', 'lvs-pussay-b_Direction (deg N)',
       'parc-du-gatinais-b_Speed(m/s)', 'parc-du-gatinais-b_Direction (deg N)',
       'arville-b_Speed(m/s)', 'arville-b_Direction (deg N)',
       'boissy-la-riviere-b_Speed(m/s)',
       'boissy-la-riviere-b_Direction (deg N)', 'angerville-1-b_Speed(m/s)',
       'angerville-1-b_Direction (deg N)', 'angerville-2-b_Speed(m/s)',
       'angerville-2-b_Direction (deg N)']

data = data[columns_use].fillna(method='ffill')
data.dropna(inplace=True)

window = 29 
lag_size = 18
lag = lag_size

series = series_to_supervised(data, window=window, lag=lag).reset_index(drop=True)
datetime_t0 = series['datetime(t)']
datetime_cols = [c for c in series.columns if 'datetime' in c]
series.drop(datetime_cols, axis=1, inplace=True)
series['datetime'] = datetime_t0
series['hour'] = series['datetime'].dt.hour
series['dayofweek'] = series['datetime'].dt.dayofweek
series['quarter'] = series['datetime'].dt.quarter
series['month'] = series['datetime'].dt.month
series['year'] = series['datetime'].dt.year 
series['dayofyear'] = series['datetime'].dt.dayofyear
series['dayofmonth'] = series['datetime'].dt.day
series['weekofyear'] = series['datetime'].dt.weekofyear

scaler = StandardScaler()

y_train = series[series.year.isin([2017,2018])][['Wind(t+18)']]
y_test = series[series.year.isin([2019])][['Wind(t+18)']]
y_holdout = series[series.year.isin([2020])][['Wind(t+18)']]

leak_cols = [c for c in series.columns if '(t+%d)' % lag_size in c]

series.drop(leak_cols, axis=1, inplace=True)

X_train = scaler.fit_transform(series[series.year.isin([2017,2018])].drop(columns=['datetime','year','month']))
X_test = scaler.transform(series[series.year.isin([2019])].drop(columns=['datetime','year','month']))
X_holdout = scaler.transform(series[series.year.isin([2020])].drop(columns=['datetime','year','month']))

X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_series = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_holdout_series = X_holdout.reshape((X_holdout.shape[0], X_holdout.shape[1], 1))

reg = keras.models.load_model("ai4impact_cnn.h5")

y_holdout_pred = reg.predict(X_holdout_series)

series = series_to_supervised(data, window=window, lag=lag).reset_index(drop=True)
datetime_t0 = series['datetime(t)']
datetime_cols = [c for c in series.columns if 'datetime' in c]
series.drop(datetime_cols, axis=1, inplace=True)
series['datetime'] = datetime_t0
series['hour'] = series['datetime'].dt.hour
series['dayofweek'] = series['datetime'].dt.dayofweek
series['quarter'] = series['datetime'].dt.quarter
series['month'] = series['datetime'].dt.month
series['year'] = series['datetime'].dt.year 
series['dayofyear'] = series['datetime'].dt.dayofyear
series['dayofmonth'] = series['datetime'].dt.day
series['weekofyear'] = series['datetime'].dt.weekofyear

holdout_result = series[series.year.isin([2020])].copy()
holdout_result['prediction'] = y_holdout_pred

pred_df = holdout_result[['datetime', 'prediction']]

agg_df = pred_df[['datetime','prediction']].groupby([pred_df['datetime'].dt.floor('H')]).agg(['mean','sum','std'])
agg_df.columns = agg_df.columns.map('_'.join)
agg_df = agg_df.reset_index()

print("predicting time")
print(round_to_hour(datetime.datetime.utcnow() - datetime.timedelta(hours=1) + datetime.timedelta(hours=18)))

current_hour = round_to_hour(datetime.datetime.utcnow() - datetime.timedelta(hours=1))

predicted_value = str(agg_df[agg_df.datetime <= current_hour].prediction_sum.values[-1])

print(predicted_value)

try:
    url = "http://3.1.52.222/submit/pred?pwd=3423549827&value="
    webUrl = urllib.request.urlopen(url + predicted_value)

    print(webUrl.read())

except:
    print("prediction not done")
