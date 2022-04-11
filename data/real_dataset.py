import os
import numpy as np
import pandas as pd
import torch
import random
import json
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import glob

# DATA_DIRS = '/mnt/infonas/data/pratham/Forecasting/DILATE'
DATA_DIRS = '/mnt/cat/data/sandy/Forecasting/'
# DATA_DIRS = '.'
def generate_train_dev_test_data(data, N_input):
    train_per = 0.6
    dev_per = 0.2
    N = len(data)

    data_train = data[:int(train_per*N)]
    data_dev = data[int(train_per*N)-N_input:int((train_per+dev_per)*N)]
    data_test = data[int((train_per+dev_per)*N)-N_input:]

    return  (data_train, data_dev, data_test)

def create_forecast_io_seqs(data, enc_len, dec_len, stride):

    data_in, data_out = [], []
    for idx in range(0, len(data), stride):
        if idx+enc_len+dec_len <= len(data):
            data_in.append(data[idx:idx+enc_len])
            data_out.append(data[idx+enc_len:idx+enc_len+dec_len])

    data_in = np.array(data_in)
    data_out = np.array(data_out)
    return data_in, data_out


def process_start_string(start_string, freq):
    '''
    Source: 
    https://github.com/mbohlkeschneider/gluon-ts/blob/442bd4ffffa4a0fcf9ae7aa25db9632fbe58a7ea/src/gluonts/dataset/common.py#L306
    '''

    timestamp = pd.Timestamp(start_string, freq=freq)
    # 'W-SUN' is the standardized freqstr for W
    if timestamp.freq.name in ("M", "W-SUN"):
        offset = to_offset(freq)
        timestamp = timestamp.replace(
            hour=0, minute=0, second=0, microsecond=0, nanosecond=0
        )
        return pd.Timestamp(
            offset.rollback(timestamp), freq=offset.freqstr
        )
    if timestamp.freq == 'B':
        # does not floor on business day as it is not allowed
        return timestamp
    return pd.Timestamp(
        timestamp.floor(timestamp.freq), freq=timestamp.freq
    )

def shift_timestamp(ts, offset):
    result = ts + offset * ts.freq
    return pd.Timestamp(result, freq=ts.freq)

def get_date_range(start_string, freq, seq_len):
    start = process_start_string(start_string, freq)
    end = shift_timestamp(start, seq_len)
    full_date_range = pd.date_range(start, end, freq=freq)
    return full_date_range


def get_list_of_dict_format(data,inject,mask):
    data_new = list()
    for entry,inj,m in zip(data,inject,mask):
        entry_dict = dict()
        entry_dict['target'] = entry
        entry_dict['target_inj']=inj
        entry_dict['target_mask']=m
        data_new.append(entry_dict)
    return data_new

def prune_dev_test_sequence(data, seq_len):
    for i in range(len(data)):
        
        data[i]['target'] = data[i]['target'][-seq_len:]
        data[i]['target_inj'] = data[i]['target_inj'][-seq_len:]
        data[i]['target_mask'] = data[i]['target_mask'][-seq_len:]
        data[i]['feats'] = data[i]['feats'][-seq_len:]
    return data


def parse_electricity(dataset_name, N_input, N_output, t2v_type=None):
    from pdb import set_trace
    #df = pd.read_csv('data/electricity_load_forecasting_panama/continuous_dataset.csv')
    df = pd.read_csv(
        os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', 'continuous_dataset.csv')
    )
    df_inject   = pd.read_csv(
        os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', 'synthesized_electricity_0.5.csv')
    )

    test_data = np.load(os.path.join(DATA_DIRS,"Outliers","Outlier","test_data.npy"))
    test_l = len(test_data)
    data = df[['nat_demand']].to_numpy().T
    data_inj = df_inject[['nat_demand']].to_numpy().T
    data_mask = df_inject[['label']].to_numpy().T
    
    # data_inj = data
    #n = data.shape[1]
    n = (1903 + 1) * 24 # Select first n=1904*24 entries because of non-stationarity in the data after first n values
    data = data[:, :n]
    data_inj = data_inj[:, :n]
    data_mask = data_mask[:, :n]
    # data_inj[...,-test_l:] = test_data 
    df = df.iloc[:n]

    # set_trace()
    units = n//N_output
    dev_len = int(0.2*units) * N_output
    test_len = int(0.2*units) * N_output
    train_len = n - dev_len - test_len

    #import ipdb ; ipdb.set_trace()

    cal_date = pd.to_datetime(df['datetime'])
    if t2v_type is None:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    elif 'mdh' in t2v_type:
        feats_date = np.stack(
            [
                cal_date.dt.month,
                cal_date.dt.day,
                cal_date.dt.hour,
            ], axis=1
        )
    elif 'idx' in t2v_type or 'local' in t2v_type:
        feats_date = np.expand_dims(np.arange(0,n), axis=-1) / n * 10.
    feats_date = np.expand_dims(feats_date, axis=0)

    feats_hod = np.expand_dims(np.expand_dims(cal_date.dt.hour.values, axis=-1), axis=0)

    #import ipdb ; ipdb.set_trace()

    feats = np.concatenate([feats_hod, feats_date], axis=-1)

    data = torch.tensor(data, dtype=torch.float)
    data_inj = torch.tensor(data_inj, dtype=torch.float)
    data_mask = torch.tensor(data_mask, dtype=torch.float)
    feats = torch.tensor(feats, dtype=torch.float)

    data_train = data[:, :train_len]
    data_inj_train = data_inj[:, :train_len]
    data_mask_train = data_mask[:, :train_len]
    feats_train = feats[:, :train_len]

    data_dev, data_test,data_inj_dev,data_inj_test,data_mask_dev,data_mask_test = [], [],[],[],[],[]
    feats_dev, feats_test = [], []
    dev_tsid_map, test_tsid_map = [], []
    for i in range(data.shape[0]):
        for j in range(train_len+N_output, train_len+dev_len+1, N_output):
            if j <= n:
                data_inj_dev.append(data_inj[i,:j])
                data_mask_dev.append(data_mask[i,:j])
                data_dev.append(data[i, :j])
                feats_dev.append(feats[i, :j])
                dev_tsid_map.append(i)
    for i in range(data.shape[0]):
        for j in range(train_len+dev_len+N_output, n+1, N_output):
            if j <= n:
                data_inj_test.append(data_inj[i,:j])
                data_mask_test.append(data_mask[i,:j])
                data_test.append(data[i, :j])
                feats_test.append(feats[i, :j])
                test_tsid_map.append(i)
    
    data_train = get_list_of_dict_format(data_train,data_inj_train,data_mask_train)
    data_dev = get_list_of_dict_format(data_dev,data_inj_dev,data_mask_dev)
    data_test = get_list_of_dict_format(data_test,data_inj_test,data_mask_test)

    for i in range(len(data_train)):
        data_train[i]['feats'] = feats_train[i]
    for i in range(len(data_dev)):
        data_dev[i]['feats'] = feats_dev[i]
    for i in range(len(data_test)):
        data_test[i]['feats'] = feats_test[i]
    feats_info = {0:(24, 16)}
    i = len(feats_info)
    for j in range(i, data_train[0]['feats'].shape[-1]):
        feats_info[j] = (-1, -1)

    seq_len = 2*N_input+N_output  #(336*2 + 168 = 840)
    data_dev = prune_dev_test_sequence(data_dev, seq_len) #54 * 840
    data_test = prune_dev_test_sequence(data_test, seq_len)
    # set_trace()
    return (
        data_train, data_dev, data_test, dev_tsid_map, test_tsid_map, feats_info
    )
