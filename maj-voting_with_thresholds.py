#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:43:02 2023
For regression model
Assign classes for each fold using the optimal threshold
Majority voting to define the final class for the average prediction
@author: Q Zeng
"""

import os
import pandas as pd
import numpy as np

root = "./eval_results_tcga-349_tumor_masked_multi-output_regression_patch"
path = os.path.join(root, 'EVAL_mo-reg_other-systemic-treatments_hcc_tumor-masked_ctranspath-tcga-paip_49_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv')
param_cl = 'Median' # Highvsrest, Lowvsrest, Median, Q1, Q3

path_thres = os.path.join(root, "EVAL_mo-reg_mondor-biopsy_hcc_tumor-masked_ctranspath-tcga-paip_157_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv")
if param_cl == 'Median' or param_cl == 'Q1' or param_cl == 'Q3':
    thres = pd.read_csv(os.path.join(path_thres, f"{param_cl.lower()}s_biopsy.csv"), header=None)
else:
    thres = pd.read_csv(os.path.join(path_thres, f"cutoffs_biopsy_{param_cl}_expv3-zscore.csv"), header=None)
print(thres)

ff = True
for fold in range(10):
    thr = thres[int(fold)].values[0]
    if np.isnan(thr):
        print('No optimal threshold!')
        continue
    else:
        print(f'Theshold for fold {fold}: {thr}')
    
    df = pd.read_csv(os.path.join(path, f"fold_{fold}.csv"), index_col=0)
    print(df.shape)
    display(df.head(3))
    
    df_pred = df.filter(df.columns[df.columns.str.contains('pred')])
    print(df_pred.shape)
    display(df_pred.head(3))
    
    # calculate average score
    if ff:
        df_results = pd.DataFrame(df_pred.T.mean(), columns=[f'pred_score_f{fold}'])
        ff = False
    else:
        df_results = df_results.join(pd.DataFrame(df_pred.T.mean(), columns=[f'pred_score_f{fold}']))
    print(df_results.shape)
    display(df_results.head(3))
    
    # assign cluster
    df_results = df_results.join(pd.DataFrame({f'pred_class_f{fold}': (df_results[f'pred_score_f{fold}'] > thr)}))
    print(df_results.shape)
    display(df_results.head(3))
    
# rename cluster
if param_cl == 'Highvsrest':
    df_results = df_results.replace({True: 'Cluster High', False: 'Cluster Median + Low'})
elif param_cl == 'Lowvsrest':
    df_results = df_results.replace({True: 'Cluster High + Median', False: 'Cluster Low'})
elif param_cl == 'Median' or param_cl == 'Q1' or param_cl == 'Q3':
    df_results = df_results.replace({True: 'High', False: 'Low'})
print(df_results.shape)
display(df_results.head(3))

df_results.index.name = 'Sample'
print(df_results.shape)
display(df_results.head(3))

df_results = df_results.join(df_results.filter(df_results.columns[df_results.columns.str.startswith('pred_class_f')]).mode(axis=1))
df_results = df_results.rename(columns = {0: 'pred_class_aver'})
if 1 in df_results.columns:
    if param_cl == 'Highvsrest':
        assert set(df_results[1].dropna()) == {'Cluster Median + Low'}
    elif param_cl == 'Lowvsrest':
        assert set(df_results[1].dropna()) == {'Cluster Low'}
    elif param_cl == 'Median' or param_cl == 'Q1' or param_cl == 'Q3':
        assert set(df_results[1].dropna()) == {'Low'}
    df_results.drop(columns=[1], inplace=True)
df_results.to_csv(os.path.join(path, f"preds_Ensembled-average_Optimal-threshold_{param_cl}_foldaverage.csv"))
