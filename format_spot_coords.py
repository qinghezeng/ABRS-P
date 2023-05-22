#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 23:03:38 2022
Format visium tissue spot coords 
@author: Q Zeng
"""
import os
import pandas as pd

# Uncomment the following two line and specify your path
# DIR_ST_OUT_SPATIAL = 'path to file tissue_positions_list.csv generated from SpaceRanger'
# DIR_SPOT_COORDS = 'save in one folder'

os.makedirs(DIR_SPOT_COORDS, exist_ok=True)

name_out = 'D2' # name of the sample

# load data
df = pd.read_csv(os.path.join(DIR_ST_OUT_SPATIAL, 'tissue_positions_list.csv'), header=None)
display(df.head(3))
print(df.shape)

# filter spots inside tissue
df = df[df[1] == 1]
df = df.reset_index(drop=True)
display(df.head(3))
print(df.shape)

# filter columns
df = df[[0, 4, 5]]
display(df.head(3))
print(df.shape)

# rename columns
df.columns = ['barcode', 'pxl_col_in_fullres', 'pxl_row_in_fullres']
display(df.head(3))
print(df.shape)

# exclude spots featuring less than 300 genes
df_spots = pd.read_csv('', index_col=0) # load the ABRS expression CSV file exported from process_st_hcc4.R line 80
display(df_spots.head(3))
print(df_spots.shape)

df = df.iloc[df.index[df.barcode.isin(df_spots.columns)],:]
display(df.head(3))
print(df.shape)

# save
df.to_csv(os.path.join(DIR_SPOT_COORDS, name_out+'.csv'), index=False)
