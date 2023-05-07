#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:32:14 2022

@author: Q Zeng
"""
from __future__ import print_function

import argparse
import torch
import os
import pandas as pd
from datasets.dataset_generic_reg import Generic_MIL_Dataset
from utils.eval_utils_reg import *

parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='root of data directory')
parser.add_argument('--data_dir', type=str, nargs="+", default=[], 
                    help='data directory')
parser.add_argument('--concat_features', action='store_true', default=False, help='enable feature concat using different extractors')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
                    help='the directory to save eval results relative to project root (default: ./eval_results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, for external validation (default: None, to use the same splits as training)')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 
                                                       "small-256", "small-512", 
                                                       "small-768", "small-1792",
                                                       "small-2048"], default='small', 
                    help='size of model, does not affect mil. Small/big is for attention backbone.')
parser.add_argument('--model_type', type=str, choices=['clam_mb_reg'], default='clam_mb_reg', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--model_activation', type=str, default=None, 
                    help='activation function after the output layer(s) to make non-negative prediction(s)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC') 
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['mo-reg_tcga_hcc_349_ABRS-score_exp_cv_00X',
                                                 'mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622',

                                                 'mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X',
                                                 'mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X',
                                                 'mo-reg_merged-mondor-resection-biopsy_hcc_382_ABRS-score_exp_cv_00X',
                                                 
                                                 'mo-reg_ABtreated-biopsy_hcc_137_ABRS-score_cv_00X',
                                                 'mo-reg_other-systemic-treatments_hcc_49_ABRS-score_cv_00X'],
                    help='indentifier for the experimental settings, see the source code for complete list')
parser.add_argument('--norm', type=str, default=None, choices=['zscore', 'scale04'],
                    help='normalization on the labels')
parser.add_argument('--test_stats', action='store_true', default=False, 
                    help='whether use test statistics to predict exact expression level')
parser.add_argument('--patch_pred', action='store_true', default=False, 
                    help='change the model arch to classify on patch instead of WSI')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoding_size = 1024

args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'norm': args.norm,
            'test_stats': args.test_stats,
            'patch_pred': args.patch_pred,
            'split': args.split,
            'save_dir': args.save_dir, 
            'concat_features': args.concat_features,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size,
            'model_activation': args.model_activation}


with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'mo-reg_tcga_hcc_349_ABRS-score_exp_cv_00X':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat=True,
                            ignore=[])
    
elif args.task == 'mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat=True,
                            ignore=[])
    
elif args.task == 'mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/mondorS2_hcc_225_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat=True,
                            ignore=[])
    
elif args.task == 'mo-reg_merged-mondor-resection-biopsy_hcc_382_ABRS-score_exp_cv_00X':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/merged-mondor-resection-biopsy_hcc_382_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat=True,
                            ignore=[])
    
elif args.task == 'mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/mondor-biopsy_hcc_157_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat=True,
                            ignore=[])
    
elif args.task == 'mo-reg_ABtreated-biopsy_hcc_137_ABRS-score_cv_00X':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/ABtreated-biopsy_hcc_137_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat=True,
                            ignore=[])
    
elif args.task == 'mo-reg_other-systemic-treatments_hcc_49_ABRS-score_cv_00X':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/other-systemic-treatments_hcc_49_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat=True,
                            ignore=[])
    
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    # Pearson corr / p across all samples for each gene
    # mean / median across all genes in the list
    # mean average / median median across 10 folds
    all_r2 = []
    all_pearson_mean = []
    all_pearson_median = []
    all_pearsons = []
    all_ppval_mean = []
    all_ppval_median = []
    all_ppvals = []
    all_MdAPE_mean = []
    all_MdAPE_median = []
    all_MdAPEs = []
    all_acc = []
    
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_smape, test_r2, test_r2s, test_pearson_mean, test_pearson_median, test_pearsons, test_ppval_mean, test_ppval_median, test_ppvals, test_MdAPE_mean, test_MdAPE_median, test_MdAPEs, df = eval(split_dataset, args, ckpt_paths[ckpt_idx])

        all_r2.append(test_r2.item())
        all_pearson_mean.append(test_pearson_mean)
        all_pearson_median.append(test_pearson_median)
        all_pearsons.append(test_pearsons)
        all_ppval_mean.append(test_ppval_mean)
        all_ppval_median.append(test_ppval_median)
        all_ppvals.append(test_ppvals)
        all_MdAPE_mean.append(test_MdAPE_mean)
        all_MdAPE_median.append(test_MdAPE_median)
        all_MdAPEs.append(test_MdAPEs)
        all_acc.append(1-test_smape)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_r2': all_r2, 
                             'test_pearson_mean': all_pearson_mean, # mean of outputs if there are multiple outputs (only 1 for ABRS)
                             'test_pearson_median': all_pearson_median, # median of outputs. same as mean for ABRS
                             'test_ppval_mean': all_ppval_mean, 
                             'test_ppval_median': all_ppval_median, 
                             'test_MdAPE_mean': all_MdAPE_mean, # MdAPE: median absolute percentage error
                             'test_MdAPE_median': all_MdAPE_median, 
                             'test_acc': all_acc})
    # List also performance for each output
    for c in range(args.n_classes):
         final_df[f'test_pearson_{c}'] = [item[c] for item in all_pearsons]
    for c in range(args.n_classes):
         final_df[f'test_ppval_{c}'] = [item[c] for item in all_ppvals]
    for c in range(args.n_classes):
         final_df[f'test_MdAPE_{c}'] = [item[c] for item in all_MdAPEs]
         
    # add rows best, mean std for each metric
    add_info = ['best']
    add_info.extend(final_df.iloc[:10, 1:4].max().tolist())
    add_info.extend(final_df.iloc[:10, 4:8].min().tolist())
    add_info.extend(final_df.iloc[:10, 8:9+args.n_classes].max().tolist())
    add_info.extend(final_df.iloc[:10, 9+args.n_classes:9+3*args.n_classes].min().tolist())
    final_df.loc[len(final_df)] = add_info
    
    add_info = ['mean']
    add_info.extend(final_df.iloc[:10, 1:].mean().tolist())
    final_df.loc[len(final_df)] = add_info
    
    add_info = ['median']
    add_info.extend(final_df.iloc[:10, 1:].median().tolist())
    final_df.loc[len(final_df)] = add_info
    
    add_info = ['std']
    add_info.extend(final_df.iloc[:10, 1:].std().tolist())
    final_df.loc[len(final_df)] = add_info

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))