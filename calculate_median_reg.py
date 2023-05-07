#!/usr/bin/env python
# coding: utf-8

## Find out the optimal thresholds on Mondor samples for regression model
### On unseen samples only if there are any

import os
import pandas as pd
import csv


root = "./eval_results_tcga-349_tumor_masked_multi-output_regression_patch"
target = "EVAL_mo-reg_mondor-biopsy_hcc_tumor-masked_ctranspath-tcga-paip_157_ABRS-score_exp_cv_00X_CLAM-MB-softplus-patch_50_s1_cv"
path = os.path.join(root, target)
folds = 10
thresh = 'median' # median, q1, q3


### Load biopsy sample list
df_biopsy = pd.read_csv('./dataset_csv/mondor-biopsy_hcc_157_ABRS_Exp.csv')
print(df_biopsy.shape)
display(df_biopsy.head(3))

list_threshs = []

for fold in range(folds):
    df = pd.read_csv(os.path.join(path, "fold_"+str(fold)+".csv"))
    df = df[df.slide_id.isin(df_biopsy.slide_id.tolist())]
    df = df.set_index('slide_id')
    df = df.filter(df.columns[df.columns.str.contains('pred')])
    df = pd.DataFrame(df.T.mean(), columns=['pred_score'])
    if thresh == 'median':
        list_threshs.append(df.pred_score.median())
    elif thresh == 'q1':
        list_threshs.append(df.quantile(q=[0.25], axis=0, numeric_only=True, interpolation='midpoint').loc[0.25, 'pred_score'])
    elif thresh == 'q3':
        list_threshs.append(df.quantile(q=[0.75], axis=0, numeric_only=True, interpolation='midpoint').loc[0.75, 'pred_score'])
    else:
        raise NotImplementedError
    print(df.shape)
    display(df.head(3))

print(list_threshs)
with open(os.path.join(path, f"{thresh}s_biopsy.csv"), 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL) 
        print
        wr.writerow(list_threshs)