import os
import numpy as np
import pandas as pd
import random
import re
import json
from pdb import set_trace
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn import metrics, datasets

DATA_DIRS = '/mnt/a99/d0/sandy/Forecasting/'


df_inject   = pd.read_csv(
    os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', '2_percent_electricity.csv')
)
# df_inject   = pd.read_csv(
#     os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', 'electricity_high_amp_train_low_amp_test.csv')
# )
filename = "electricity"
data_mask = df_inject['label'].to_numpy()  #### original anomalies
n = (1903 + 1) * 24 # Select first n=1904*24 entries because of non-stationarity in the data after first n values
data_mask = data_mask[:n]
test_data = np.load(os.path.join(DATA_DIRS,'data','electricity',"electricity_mask_roc.npy"))
test_l = len(test_data)
fpr, tpr, thresholds = roc_curve(data_mask[200:200+test_l], test_data)
score = metrics.auc(fpr, tpr)
cm = confusion_matrix(data_mask[...,200:200+test_l].reshape(-1,), test_data)
dic = dict({'fpr':list(fpr),'tpr':list(tpr),'cm':list(cm)})

json_obj = json.dumps(dic,indent=8)
with open(f"Results/{filename}_roc.json",'w+') as files:
    files.write(json_obj)
