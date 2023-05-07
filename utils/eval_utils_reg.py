#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:41:09 2022

@author: Q Zeng
"""
import numpy as np

import torch
import torch.nn.functional as F
from models.model_clam_reg import CLAM_MB_reg
import os
import pandas as pd
from utils.utils_reg import *
from utils.core_utils_reg import Accuracy_Logger

from torchmetrics.functional import r2_score
from scipy.stats import pearsonr
from torchmetrics.functional import symmetric_mean_absolute_percentage_error
import statistics
import json

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None:
        model_dict.update({"size_arg": args.model_size})
        
    if args.model_activation is not None:
        model_dict.update({"activation": args.model_activation})
        
    print(model_dict)
    
    if args.model_type == 'clam_mb_reg':
        model = CLAM_MB_reg(**model_dict)
    else:
        raise NotImplementedError

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_smape, test_r2, test_r2s, test_pearsons, test_ppvals, test_MdAPEs, df, _ = summary(model, loader, args)
    test_pearson_mean = sum(test_pearsons)/len(test_pearsons)
    test_ppval_mean = sum(test_ppvals)/len(test_ppvals)
    test_MdAPE_mean = sum(test_MdAPEs)/len(test_MdAPEs)
    test_pearson_median = statistics.median(test_pearsons)
    test_ppval_median = statistics.median(test_ppvals)
    test_MdAPE_median = statistics.median(test_MdAPEs)
    print('Test SMAPE: {:.4f}, r2: {:.4f}, mean pearson: {:.4f}, median pearson: {:.4f}, mean ppval: {:.4f}, median ppval: {:.4f}, mean MdAPE: {:.4f}, median MdAPE: {:.4f}'.format(test_smape, test_r2, test_pearson_mean, test_pearson_median, test_ppval_mean, test_ppval_median, test_MdAPE_mean, test_MdAPE_median))
    return model, patient_results, test_smape, test_r2, test_r2s, test_pearson_mean, test_pearson_median, test_pearsons, test_ppval_mean, test_ppval_median, test_ppvals, test_MdAPE_mean, test_MdAPE_median, test_MdAPEs, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes, reg=True)
    model.eval()
    
    test_smape = 0.
    r2s = []
    pearsons = []
    ppvals = []
    MdAPEs = []

    all_preds = np.zeros((len(loader), args.n_classes), dtype=np.float64)
    all_targets = np.zeros((len(loader), args.n_classes), dtype=np.float64)

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if args.patch_pred:
                _, A_raw, results_dict = model(data, patch_pred=args.patch_pred) # A_raw: (1, N) for sb, (k, N) for mb
                patch_pred = results_dict['patch_pred'] # (N, k) for mb
                patch_pred = torch.transpose(patch_pred, 1, 0) # (k, N)
                logits = patch_pred * F.softmax(A_raw, dim=1) # (k, N)
                logits = torch.transpose(torch.sum(logits, dim=1, keepdim=True), 1, 0) # (k, 1) -> (1, k)
            else:
                logits, _, _ = model(data)
            
        if args.norm == 'zscore':
            if batch_idx == 0:
                print('Reverse zscore on the predicted values...')
                if args.test_stats:
                    mean_std = json.load(open(f'dataset_csv/{os.path.basename(os.path.splitext(loader.dataset.csv_path)[0])}_mean-std.json'))
                else:
                    slist = args.models_exp_code.split('_')
                    mean_std = json.load(open(f"dataset_csv/{slist[1]}_{slist[2]}_{slist[5]}_{slist[6]}_{slist[7].replace('exp', 'Exp').replace('log2cpm', 'log2CPM')}_mean-std.json"))
            for c in range(args.n_classes):
                original_mean, original_std = mean_std[loader.dataset.label_col[c]]
                logits[0, c] = logits[0, c]*original_std + original_mean
        elif args.norm == 'scale04':
            if batch_idx == 0:
                print('Reverse scale04 on the predicted values...')
                from sklearn.externals import joblib
                if args.test_stats:
                    scaler = joblib.load(f'dataset_csv/{os.path.basename(os.path.splitext(loader.dataset.csv_path)[0])}_scaler04.save')
                else:
                    slist = args.models_exp_code.split('_')
                    mean_std = json.load(open(f"dataset_csv/{slist[1]}_{slist[2]}_{slist[5]}_{slist[6]}_{slist[7].replace('exp', 'Exp').replace('log2cpm', 'log2CPM')}_scaler04.save"))
            logits = logits.cpu().detach().numpy().reshape(1, -1)
            logits = torch.FloatTensor(scaler.inverse_transform(logits)).to(device)
        else:
            pass
        
        acc_logger.log_reg(logits, targets)
        
        all_preds[batch_idx] = logits.cpu().numpy()
        all_targets[batch_idx] = targets.cpu().numpy()
        
        patient_results.update({'slide_id': np.array(slide_id), 'pred': logits, 'targets': targets.cpu().numpy()})
        
        smape = symmetric_mean_absolute_percentage_error(logits, targets)
        test_smape += smape.cpu().numpy()

        del data
    test_smape /= len(loader)

    r2 = r2_score(torch.FloatTensor(all_preds), torch.FloatTensor(all_targets))
    
    results_dict = {'slide_id': slide_ids}
    for c in range(args.n_classes):
        r2s.append(r2_score(torch.FloatTensor(all_preds[:, c]), torch.FloatTensor(all_targets[:, c])))
        pearson, ppval = pearsonr(all_preds[:, c], all_targets[:, c])
        pearsons.append(pearson)
        ppvals.append(ppval)
        # MdAPE: median absolute percentage error
        MdAPEs.append(statistics.median(np.absolute(all_targets[:, c] - all_preds[:, c]) / np.absolute(all_targets[:, c])))
        results_dict.update({'target_{}'.format(c): all_targets[:,c], 'pred_{}'.format(c): all_preds[:,c]})

    df = pd.DataFrame(results_dict)
    
    return patient_results, test_smape, r2, r2s, pearsons, ppvals, MdAPEs, df, acc_logger
