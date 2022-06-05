import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name',  type=str, help='Dataset name', default="electricity")
args = parser.parse_args()
path_dir = f"Results/{args.dataset_name}_train_m/"
dirs = os.listdir(path_dir)
final_input,final_25,final_50 = [], [],[]
for folder in dirs:
    path = os.path.join(path_dir,folder,args.dataset_name)
    # print(path)
    preds = np.load(path+'/trans-mse-ar_pred_mu.npy')
    inputs = np.load(path+'/inputs.npy')
    trues = np.load(path+'/targets.npy')
    new_preds = preds.squeeze()
    new_trues = trues.squeeze()
    N_input = 336
    # print(new_preds.shape,new_trues.shape)
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
        msep = np.quantile(mse,0.75)+iqr(mse,interpolation='midpoint')*1.5
        if np.max(line['mse'])>msep:
            # print(np.argsort(line['mse'])[-1],line['mse']>msep,max(line['mse']))
            for x in range(1,3):
                i = np.argsort(line['mse'])[-x]
                if line['mse'][i]>msep:
                    final[j]['mask'][i]= 1
    mask = final[0]['mask'].flatten()
    for line in final[1:]:
        mask=np.hstack((mask,line['mask'][1:].flatten()))
    if "25" in path:
        final_25 = mask 
    else:
        final_50 = mask
# print(final_25.shape,final_50.shape)
length = min(len(final_25)-25,len(final_50))
final_25 = final_25[25:length+25]
final_50 = final_50[:length]
final_mask = final_25*final_50
# print(final_mask.shape)

cwd = "/mnt/cat/data/sandy/Forecasting/data/"
save_path=os.path.join(cwd,args.dataset_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(os.path.join(cwd,f"{args.dataset_name}_mask_iqr.npy"),final_mask)
print(f"File saved as {cwd}{args.dataset_name}_mask_iqr.npy !!")



