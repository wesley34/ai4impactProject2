import schedule
import time

import io
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import requests
import pytz, datetime
import time
import os 

from googletrans import Translator
translator = Translator()

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

def crawl_data_from_RTE():

    required_columns = ['Périmètre', 'Nature', 'Date', 'Heures', 'Consommation', 'Thermique','Eolien']

    url_list = ["https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_Annuel-Definitif_2017.zip",
                "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_Annuel-Definitif_2018.zip",
                "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_En-cours-Consolide.zip",
            "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_En-cours-TR.zip"]

    df_list = []

    for url in url_list:
        content = requests.get(url)
        zf = ZipFile(BytesIO(content.content))

        for item in zf.namelist():
            print("File in zip: "+  item)

        # find the first matching csv file in the zip:
        match = [s for s in zf.namelist() if ".xls" in s][0]
        
        tmp_df = pd.read_table(zf.open(match), index_col=False, usecols = required_columns, encoding='ISO-8859-1').head(-1)
        
        df_list.append(tmp_df)
        
    df = pd.concat(df_list).reset_index(drop=True)

    translated_columns = [translator.translate(i, src='fr', dest='en').text for i in df.columns]
    df.columns = translated_columns

    local = pytz.timezone ("Europe/Paris")

    df['datetime'] = df['Dated'] + " " + df['Hours']

    df['datetime'] = df['datetime'].apply(lambda x: local.localize(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M"), is_dst=True).astimezone(pytz.utc))

    df = df.drop_duplicates(subset=['datetime'])
    df['datetime'] = df['datetime'].astype(str).apply(lambda x: x.split("+")[0])
    df['datetime'] = pd.to_datetime(df['datetime'])

    df = df[df['Consumption'] != "ND"]

    for i in ['Consumption', 'Thermal', 'Wind', 'Solar', 'Hydraulic', 'Bioenergies','Ech. physical']:
        df[i] = pd.to_numeric(df[i]) * 250

    wf_list = ["guitrancourt", "lieusaint", "lvs-pussay", "parc-du-gatinais", "arville", "boissy-la-riviere", "angerville-1", "angerville-2",
    "guitrancourt-b", "lieusaint-b", "lvs-pussay-b", "parc-du-gatinais-b", "arville-b", "boissy-la-riviere-b", "angerville-1-b", "angerville-2-b"]

    forecast_df_list = []

    for forecast in wf_list:
        
        hist_url = "https://ai4impact.org/P003/historical/" + forecast +".csv"

        r = requests.get(hist_url)
        data = r.content.decode('utf8').split("UTC\n")[1]
        hist_tmp_df = pd.read_csv(io.StringIO(data))
        
        current_url = "https://ai4impact.org/P003/" + forecast +".csv"

        r = requests.get(current_url)
        data = r.content.decode('utf8').split("UTC\n")[1]
        current_tmp_df = pd.read_csv(io.StringIO(data))
        
        tmp_df = pd.concat([hist_tmp_df,current_tmp_df]).reset_index(drop=True).rename(columns={'Speed(m/s)':forecast + '_Speed(m/s)', 'Direction (deg N)':forecast + '_Direction (deg N)'})
        
        tmp_df['datetime'] = pd.to_datetime(tmp_df['Time'].str.replace("UTC", ""))
        
        tmp_df = tmp_df.drop(columns=['Time'])
        
        forecast_df_list.append(tmp_df)
        
    main_df = df.copy()

    for i in forecast_df_list:
        main_df = main_df.merge(i, how='left', on='datetime')
    
    main_df.to_csv("combined_energy_data.csv", index=False)

    os.system('gsutil cp combined_energy_data.csv gs://ai4impact-hkdragons')

    print("finished crawling and export to gcs")

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


def predict_and_submit():
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


#schedule.every(1).minutes.do(crawl_data_from_RTE)
#schedule.every(1).minutes.do(predict_and_submitt)

while 1:
    #schedule.run_pending()
    try:
        crawl_data_from_RTE()
    except:
        pass
    try:
        predict_and_submit()
    except:
        pass

    time.sleep(1)
