#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:09:11 2022
Average the predictions from the 10 fold CLAM models
@author: Q Zeng
"""

import os
import pandas as pd
import itertools

path = PATH_TO_RESULTS

for fold in range(10):
    df = pd.read_csv(os.path.join(os.path.join(path, f'fold_{fold}.csv')))
    df.set_index('slide_id', inplace=True)
    print(df.shape)
    display(df.head(3))

    # filter the predictions and rename the cols with fold name
    df_pred_ = df.filter(df.columns[df.columns.str.contains('pred')])
    df_pred_.columns = [ncol+f'_{fold}' for ncol in df_pred_.columns]
    print(df_pred_.shape)
    display(df_pred_.head(3))
    
    if fold == 0:
        # filter the targets
        df_target = df.filter(df.columns[df.columns.str.contains('target')])
        print(df_target.shape)
        display(df_target.head(3))
        
        df_pred = df_pred_
    else:
        # concat the predictions from all folds
        df_pred = pd.concat([df_pred, df_pred_], axis=1)
    print(df_pred.shape)
    display(df_pred.head(3))

# average for each gene
dicts = {}
for i in range(df_target.shape[1]):
        dicts[f'aver_pred_{i}'] = df_pred.loc[:, df_pred.columns.str.startswith(f'pred_{i}')].T.mean()
print(dicts)
df_results = pd.DataFrame(dicts)
print(df_results.shape)
display(df_results.head(3))

# rebuild the df by inserting the target and average for each gene
df_results = pd.concat([df_results, df_target], axis=1)
df_results = df_results[list(itertools.chain(*zip([f'target_{i}' for i in range(df_target.shape[1])], [f'aver_pred_{i}' for i in range(df_target.shape[1])])))]
df_results.to_csv(os.path.join(path, 'fold_average.csv'))

