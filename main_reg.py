#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:16:54 2022

@author: Q Zeng
"""

from __future__ import print_function
# internal imports
from utils.file_utils import save_pkl
from utils.core_utils_reg import train
from datasets.dataset_generic_reg import Generic_MIL_Dataset

import argparse
import os
import torch
import pandas as pd
import numpy as np
import time
import wandb


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_r2 = []
    all_val_r2 = []
    all_test_pearson = []
    all_val_pearson = []
    all_test_ppval = []
    all_val_ppval = []
    all_test_acc = []
    all_val_acc = []
    
    train_times = 0.
    
    folds = np.arange(start, end)
    for i in folds:
        tags = []
        if args.extractor_model:
            tags.append(args.extractor_model)
        if args.model_activation:
            job_type = args.model_activation
        else:
            job_type = 'wo_activation'
        run = wandb.init(project=args.project_name, group=args.task, 
                         job_type=job_type, name=f'fold{i}', tags=tags, entity="hezaii", reinit=True)
        wandb.config = settings

        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        print(len(train_dataset))
        
        datasets = (train_dataset, val_dataset, test_dataset)

        train_time_elapsed = -1 # Default time

        results, test_r2, val_r2, test_pearson, val_pearson, test_ppval, val_ppval, test_acc, val_acc, train_time_elapsed  = train(datasets, i, args)
        print("Training time in s for fold {}: {}".format(i, train_time_elapsed))

        all_test_r2.append(test_r2.item())
        all_val_r2.append(val_r2.item())
        all_test_pearson.append(test_pearson)
        all_val_pearson.append(val_pearson)
        all_test_ppval.append(test_ppval)
        all_val_ppval.append(val_ppval)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

        run.finish()
        
        train_times += train_time_elapsed
    print()
    print("Average train time in s per fold: {}".format(train_times / len(folds)))
    # print used gpu which could affect training time
    print('Used GPU: {}, ({})'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
    print()

    final_df = pd.DataFrame({'folds': folds, 'test_r2': all_test_r2, 'val_r2': all_val_r2, 
                             'test_pearson': all_test_pearson, 'val_pearson': all_val_pearson,
                             'test_ppval': all_test_ppval, 'val_ppval': all_val_ppval,
                             'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='root of data directory')
parser.add_argument('--data_dir', type=str, nargs="+", default=[], 
                    help='data directory')
parser.add_argument('--concat_features', action='store_true', default=False, help='enable feature concat using different extractors')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['mse'], default='mse',
                     help='slide-level classification loss function (default: mse)')
parser.add_argument('--model_type', type=str, choices=['clam_mb_reg'], default='clam_mb_reg', 
                    help='type of model (default: clam_mb_reg, clam w/ multiple attention branch for regression)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'small-768',
                                                       'big'], default='small', 
                    help='size of model, does not affect mil')
parser.add_argument('--model_activation', type=str, default=None, help='activation function after the output layer(s) to make non-negative prediction(s)')
parser.add_argument('--task', type=str, choices=['mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622'])
parser.add_argument('--use_h5', action='store_true', default=False, help='load features from h5 format')
parser.add_argument('--train_augm', action='store_true', default=False, help='enable data augmentation on training')
parser.add_argument('--project_name', type=str, default='clam_hcc', help='project name for wandb')
parser.add_argument('--extractor_model', type=str, default=None, help='model used for feature extraction')
parser.add_argument('--patch_pred', action='store_true', default=False, 
                    help='change the model arch to classify on patch instead of WSI')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            'model_activation': args.model_activation,
            'concat_features': args.concat_features,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'train_augm': args.train_augm,
            'patch_pred': args.patch_pred,
            'opt': args.opt}

print('\nLoad Dataset')

start_time = time.time()
if args.task == 'mo-reg_tcga_hcc_349_ABRS-score_exp_cv_622':
    args.n_classes=1
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_hcc_349_ABRS-score_Exp.csv',
                            data_dir= args.data_dir,
                            concat_features = args.concat_features,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {},
                            label_col = ["ABRS"],
                            patient_strat= True,
                            ignore=[])
    
else:
    raise NotImplementedError

time_elapsed = time.time() - start_time
print("load dataset took {} seconds".format(time_elapsed))

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


