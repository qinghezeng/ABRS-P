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
score_type = "weighted_pred" #attention
norm = 'rescale01' # percentile-rescale01
sname = 'D7'

df_st = pd.read_csv(os.path.join(path1, f'{sname}_ABRS.csv'), index_col=0)
print(df_st.shape)
display(df_st.head(3))

df_att = pd.read_csv(os.path.join(path2, f'ensembled-aver_{score_type}_scores_10f_{norm}', f'{sname}_rot90_0.csv'))
print(df_att.shape)
display(df_att.head(3))

sort_index = np.argsort(df_att.weighted_pred_score)  # from small to great

df_low = df_st.filter(df_att.barcode[sort_index[:n_patch]]).T
df_high = df_st.filter(df_att.barcode[sort_index[-n_patch:]]).T

df_att[f'G{n_patch}'] = 'Non_Info'
df_att.loc[sort_index[:n_patch], f'G{n_patch}'] = 'Low'
df_att.loc[sort_index[-n_patch:], f'G{n_patch}'] = 'High'

df_att.to_csv(os.path.join(path2, f'ensembled-aver_{score_type}_scores_10f_{norm}', f'{sname}_rot90_G{n_patch}.csv'))
