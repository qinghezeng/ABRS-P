#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:10:23 2022
Evaluate ensembled average predictions
@author: Q Zeng
"""
from __future__ import print_function

import numpy as np

import argparse
import torch
import os
import pandas as pd

from utils.utils_reg import *

from torchmetrics.functional import r2_score
from scipy.stats import pearsonr
from torchmetrics.functional import symmetric_mean_absolute_percentage_error
import statistics

parser = argparse.ArgumentParser(description='CLAM Ensembled Average Evaluation Script')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
                    help='the directory to save eval results relative to project root (default: ./eval_results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))
assert os.path.isdir(args.save_dir)

if __name__ == "__main__":
    # Pearson corr / p across all samples for each gene
    # mean / median across all genes in the list

    r2s = []
    pearsons = []
    ppvals = []
    MdAPEs = []
    
    df_data = pd.read_csv(os.path.join(args.save_dir, 'fold_average.csv'), index_col=0)
    # display(df_data.head(3))
    
    # filter targets
    df_target = df_data.filter(df_data.columns[df_data.columns.str.contains('target')])
    # display(df_target.head(3))
    
    # filter predictions
    df_pred = df_data.filter(df_data.columns[df_data.columns.str.contains('pred')])
    # display(df_pred.head(3))
    
    smape = symmetric_mean_absolute_percentage_error(torch.tensor(df_pred.values), torch.tensor(df_target.values))
    acc = 1 - smape.cpu().numpy()

    r2 = r2_score(torch.tensor(df_pred.values), torch.tensor(df_target.values))
    r2 = r2.cpu().numpy()
    
    for c in range(df_target.shape[1]):
        r2s.append(r2_score(torch.FloatTensor(df_pred.iloc[:, c]), torch.FloatTensor(df_target.iloc[:, c])).cpu().numpy())
        pearson, ppval = pearsonr(df_pred.iloc[:, c], df_target.iloc[:, c])
        pearsons.append(pearson)
        ppvals.append(ppval)
        # MdAPE: median absolute percentage error
        MdAPEs.append(statistics.median(np.absolute(df_target.iloc[:, c] - df_pred.iloc[:, c]) / np.absolute(df_target.iloc[:, c])))
    
    final_df = pd.DataFrame.from_dict({'test_acc': acc,
                                       'test_r2': r2, 
                                       'test_r2_mean': sum(r2s) / len(r2s), 
                                       'test_r2_median': statistics.median(r2s),
                                       'test_pearson_mean': sum(pearsons) / len(pearsons), # mean of outputs if there are multiple outputs (only 1 for ABRS)
                                       'test_pearson_median': statistics.median(pearsons), # median of outputs. same as mean for ABRS
                                       'test_ppval_mean': sum(ppvals) / len(ppvals), 
                                       'test_ppval_median': statistics.median(ppvals), 
                                       'test_MdAPE_mean': sum(MdAPEs) / len(MdAPEs), 
                                       'test_MdAPE_median': statistics.median(MdAPEs)}, 'index')
    
    for c in range(df_target.shape[1]):
        final_df.loc[f'test_r2_{c}'] = r2s[c]
    for c in range(df_target.shape[1]):
        final_df.loc[f'test_pearson_{c}'] = pearsons[c]
    for c in range(df_target.shape[1]):
        final_df.loc[f'test_ppval_{c}'] = ppvals[c]
    for c in range(df_target.shape[1]):
        final_df.loc[f'test_MdAPE_{c}'] = MdAPEs[c]

    final_df.to_csv(os.path.join(args.save_dir, 'summary_average.csv'))
