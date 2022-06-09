import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn import metrics, datasets
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name',  type=str, help='Dataset name', default="electricity")
parser.add_argument('--N_input',  type=int, help='Input length for forecasting model', default=-1)
args = parser.parse_args()
path_dir = f"Results/{args.dataset_name}_train_m/"
dirs = os.listdir(path_dir)
final_input,final_25,final_50 = [], [],[]

DATA_DIRS = "/mnt/a99/d0/sandy/Forecasting/"
df = pd.read_csv(
    os.path.join(DATA_DIRS, 'data', 'electricity_load_forecasting_panama', '2_percent_electricity.csv')
)
data = df[['nat_demand']].to_numpy().T
# labels =  df[['label']].to_numpy()





for folder in dirs:
    labels =  df[['label']].to_numpy()
    path = os.path.join(path_dir,folder,args.dataset_name)
    # print(path)
    preds = np.load(path+'/trans-mse-ar_pred_mu.npy')
    inputs = np.load(path+'/inputs.npy')
    trues = np.load(path+'/targets.npy')
    new_preds = preds.squeeze()
    new_trues = trues.squeeze()
    if args.N_input == -1:
        args.N_input = 336
    # print(new_preds.shape,new_trues.shape)
    n = (args.N_input-1)//50 + 1
    (l,h) = new_preds.shape
    mse = [mean_squared_error(i,j) for i,j in zip(new_preds,new_trues)]


    minmse = min(mse)
    maxmse = max(mse)
    mse_norm = [(m-minmse)/(maxmse-minmse) for m in mse]
    final = []
    for line in range(0,l,n-1):
        chunk = dict()
        chunk['preds'] = new_preds[line:line+n]
        chunk['trues'] = new_trues[line:line+n]
        chunk['mask'] = np.zeros_like(new_trues[line:line+n])
        chunk['mse'] = mse_norm[line:line+n]
    final.append(chunk)
    

    labels = labels[200:200+l*h].reshape(-1,h)


    last = 3
    # if h==25:
    #     last = 3

    label = [1 if sum(line)>5 else 0 for line in labels]

    fpr, tpr, thresholds = roc_curve(label, mse_norm)
    score = metrics.auc(fpr, tpr)
    # print(labels.shape,new_preds.shape,len(label),len(mse))
    # print(thresholds)
    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 6)
    gmeanOpt = round(gmean[index], ndigits = 6)
    fprOpt = round(fpr[index], ndigits = 6)
    tprOpt = round(tpr[index], ndigits = 6)
    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

    pred_labels = [1 if i>thresholdOpt else 0 for i in mse_norm ]


    for j,line in enumerate(final):
        # print(sorted(line['mse'],reverse=True))
        # msep = np.quantile(mse,0.75)+iqr(mse,interpolation='midpoint')*1.5
        
        # print(np.argsort(line['mse'])[-1],line['mse']>msep,max(line['mse']))
        for x in range(1,3):
            if len(line['mse'])<2:
                continue
            i = np.argsort(line['mse'])[-x]
            
            if line['mse'][i]>thresholdOpt:
                final[j]['mask'][i]= 1
    mask = final[0]['mask'].flatten()
    for line in final[1:]:
        mask=np.hstack((mask,line['mask'][1:].flatten()))
    if "25" in path:
        final_25 = mask 
    else:
        final_50 = mask
# print(final_25.shape,final_50.shape)
length = min(len(final_25),len(final_50))
final_25 = final_25[:length]
final_50 = final_50[:length]
final_mask = final_25*final_50
# print(final_mask.shape)

cwd = "/mnt/cat/data/sandy/Forecasting/data/"
save_path=os.path.join(cwd,args.dataset_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
file_p = os.path.join(save_path,f"{args.dataset_name}_mask_roc.npy")
np.save(file_p,final_mask)
print(f"File saved as {save_path}/{args.dataset_name}_mask_roc.npy !!")
import subprocess

subprocess.run(["scp",file_p, f"sandy@dog.cse.iitb.ac.in:/mnt/a99/d0/sandy/Forecasting/data/{args.dataset_name}/{args.dataset_name}_mask_iqr.npy"])
print(f"File saved as sandy@dog.cse.iitb.ac.in:/mnt/a99/d0/sandy/Forecasting/data/{args.dataset_name}/{args.dataset_name}_mask_iqr.npy !!")

