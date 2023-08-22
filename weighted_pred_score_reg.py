#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:16:15 2023

Weighted pred map for regression
modified on attention_score_reg which input the features and output the raw attention scores

@author: Q Zeng
"""

from __future__ import print_function

import argparse
import torch
import os
import time
from utils.utils_reg import *
from datasets.dataset_generic_reg import Generic_MIL_Dataset
from utils.eval_utils_reg import *

parser = argparse.ArgumentParser(description='CLAM Weighted Prediction Score Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--data_dir', type=str, nargs="+", default=[], 
                    help='data directory')
parser.add_argument('--concat_features', action='store_true', default=False, help='enable feature concat using different extractors')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--eval_dir', type=str, default='./eval_results',
					help='directory to save eval results')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big', 
                                                       "small-256", "small-512", 
                                                       "small-768", "small-1792",
                                                       "small-2048"], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_activation', type=str, default=None, help='activation function after the output layer(s) to make non-negative prediction(s)')
parser.add_argument('--model_type', type=str, choices=['clam_mb_reg'], default='clam_mb_reg', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622',
                                                 'mo-reg_mondorS2_hcc_225_ABRS-score_exp_cv_00X',
                                                 'mo-reg_mondor-biopsy_hcc_157_ABRS-score_exp_cv_00X',
                                                 'mo-reg_st_hcc_4_ABRS-score_cv_00X'],
                    help='indentifier for the experimental settings, see the source code for complete list')

parser.add_argument('--feature_bags', type=str, nargs='+', default=None, 
                    help='names of patched feature files (ends with .pt) for visualization (default: [])')
args = parser.parse_args()

encoding_size = 1024

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
    
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]

args.save_dir = os.path.join(args.eval_dir, 'EVAL_' + str(args.save_exp_code))

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'concat_features': args.concat_features,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size,
            'model_activation': args.model_activation}

print(settings)

if args.task == 'mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
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
                            patient_strat= True,
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
                            patient_strat= True,
                            ignore=[])
    
elif args.task == 'mo-reg_st_hcc_4_ABRS-score_cv_00X':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/st_hcc_4_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])

else:
    raise NotImplementedError

if __name__ == "__main__":
     print("dataset:")
     print(dataset)
     print("data_dir:")
     print(args.data_dir)
     
     for ckpt_idx in range(len(ckpt_paths)):
         save_dir = args.save_dir
         save_dir = os.path.join(save_dir, "weighted_patch_pred_scores_" + str(folds[ckpt_idx]))
         os.makedirs(save_dir, exist_ok=True)
         
         model = initiate_model(args, ckpt_paths[ckpt_idx])
         
         if args.feature_bags is not None:
             feature_bags = args.feature_bags
         else:
             feature_bags = []
             for data_dir_ in args.data_dir:
                 feature_bags.extend(os.listdir(data_dir_))
             feature_bags = sorted(list(set(feature_bags)))
             feature_bags = [features for features in feature_bags if features.endswith(".pt")]
         
         total = len(feature_bags)
         times = 0.
         
         for i in range(total): 
             print("\n\nprogress: {:.2f}, {}/{} in current model. {} out of {} models".format(i/total, i, total, ckpt_idx, len(ckpt_paths)))
             print('processing {}'.format(feature_bags[i]))
    
             if args.concat_features:
                bag_features = []
                for data_dir_ in args.data_dir:
                    if os.path.isfile(os.path.join(data_dir_,feature_bags[i])):
                        bag_features.append(torch.load(os.path.join(data_dir_, feature_bags[i]), map_location=lambda storage, 
                                                       loc: storage.cuda(0)))

                    else:
                        raise FileNotFoundError("Please check the data_dir!")
                bag_features = torch.cat(bag_features, 1)
             else:
                find = False
                for data_dir_ in args.data_dir:
                    if os.path.isfile(os.path.join(data_dir_,feature_bags[i])):
                        bag_features = torch.load(os.path.join(data_dir_, feature_bags[i]), map_location=lambda storage, 
                                                  loc: storage.cuda(0))
                        find = True
                        break
                if not find:
                    raise FileNotFoundError("Please check the data_dir!")
             
             time_elapsed = -1
             start_time = time.time()

             _, A_raw, results_dict = model(bag_features, patch_pred=True) # A_raw: (1, N) for sb, (k, N) for mb
             patch_pred = results_dict['patch_pred'] # (N, 2) for sb, (N, k) for mb
             
             patch_pred = torch.transpose(patch_pred, 1, 0) # (k, N)
             weighted_patch_pred = patch_pred * A_raw # (k, N)
                 
             time_elapsed = time.time() - start_time
             times += time_elapsed
             
             print("Patch pred")
             print(patch_pred.size())
             print("Max:")
             print(torch.max(patch_pred))
             print("Min:")
             print(torch.min(patch_pred))
             torch.save(patch_pred, os.path.join(save_dir, "patch_pred_score_" + feature_bags[i]))
             
             print("Weighted patch pred")
             print(weighted_patch_pred.size()) # torch.Size([1, 22857])
             print("Max:")
             print(torch.max(weighted_patch_pred))
             print("Min:")
             print(torch.min(weighted_patch_pred))
             torch.save(weighted_patch_pred, os.path.join(save_dir, "weighted_patch_pred_score_" + feature_bags[i]))
    
         times /= total
    
         print("average time in s per slide: {}".format(times))
