import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
path = "Results/gecco/gecco_ip_100_op_50_robust_1/gecco/"
preds = np.load(path+'trans-mse-ar_pred_mu.npy')
inputs = np.load(path+'inputs.npy')
trues = np.load(path+'targets.npy')
new_preds = preds.squeeze()
new_trues = trues.squeeze()
N_input = 480
n = (N_input-1)//50 + 1
l = new_preds.shape[0]
final = []
for line in range(0,l,n-1):
    chunk = dict()
    chunk['preds'] = new_preds[line:line+n]
    chunk['trues'] = new_trues[line:line+n]
    chunk['mask'] = np.zeros_like(new_trues[line:line+n])
    chunk['mse'] = [mean_squared_error(i,j) for i,j in zip(new_preds[line:line+n],new_trues[line:line+n])]
    final.append(chunk)
mse = [mean_squared_error(i,j) for i,j in zip(new_preds,new_trues)]
for j,line in enumerate(final):
    # print(sorted(line['mse'],reverse=True))
    if True in list((np.quantile(mse,0.75)+stats.iqr(mse,interpolation='midpoint')*1.5)<line['mse']):
        # print(np.argsort(line['mse'])[-1])
        i = np.argsort(line['mse'])[-1]
        final[j]['mask'][i]=1
mask = final[0]['mask'].flatten()
for line in final[1:]:
    mask=np.hstack((mask,line['mask'][1:].flatten()))
np.save("/mnt/a99/d0/sandy/Forecasting/data/water_quality/mask_gecco.npy",mask)