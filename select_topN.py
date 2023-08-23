#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:04:03 2023

@author: Q Zeng
"""

import os
import pandas as pd
import numpy as np

path1 = "PATH_ST"
path2 = 'PATH_EVAL_ST'

n_patch = 100
score_type = "weighted_patch_pred" #attention
norm = 'rescale01' # percentile-rescale01
sname = 'D7'

df_st = pd.read_csv(os.path.join(path1, f'{sname}_ABRS.csv'), index_col=0)
print(df_st.shape)
display(df_st.head(3))

df_pred = pd.read_csv(os.path.join(path2, f'ensembled-aver_{score_type}_scores_10f_{norm}', f'{sname}_rot90_0.csv'))
print(df_pred.shape)
display(df_pred.head(3))

sort_index = np.argsort(df_pred[f'{score_type}_score'])  # from small to great

df_low = df_st.filter(df_pred.barcode[sort_index[:n_patch]]).T
df_high = df_st.filter(df_pred.barcode[sort_index[-n_patch:]]).T

df_pred[f'G{n_patch}'] = 'Non_Info'
df_pred.loc[sort_index[:n_patch], f'G{n_patch}'] = 'Low'
df_pred.loc[sort_index[-n_patch:], f'G{n_patch}'] = 'High'

df_pred.to_csv(os.path.join(path2, f'ensembled-aver_{score_type}_scores_10f_{norm}', f'{sname}_rot90_G{n_patch}.csv'))
