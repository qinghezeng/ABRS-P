#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:37:34 2022

@author: Q Zeng
"""

import numpy as np
import torch
import torch.nn as nn
from utils.utils_reg import *
import os
from datasets.dataset_generic_reg import save_splits
from models.model_clam_reg import CLAM_MB_reg

import time
import wandb
import torch.nn.functional as F

from torchmetrics.functional import r2_score
from scipy.stats import pearsonr
from torchmetrics.functional import symmetric_mean_absolute_percentage_error

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes, reg):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        if reg:
            self.initialize_reg()
        else:
            self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count
    
    def initialize_reg(self):
        self.logits = np.zeros((0, self.n_classes), dtype=np.float64)
        self.targets = np.zeros((0, self.n_classes), dtype=np.float64)
    
    def log_reg(self, logits, targets):
        self.logits = np.concatenate((self.logits, logits.cpu().detach().numpy()))
        self.targets = np.concatenate((self.targets, targets.cpu().detach().numpy()))
    
    def get_summary_reg(self, c): # per class peason correlation and r2score
        count = self.logits.shape[0]
        
        if count == 0: 
            r2 = None
            pearson = None
            ppval = None
        else:
            r2 = r2_score(torch.FloatTensor(self.logits[:, c]), torch.FloatTensor(self.targets[:, c]))
            pearson, ppval = pearsonr(self.logits[:, c], self.targets[:, c])
        
        return r2, pearson, ppval, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch = -1

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.best_epoch = epoch
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    start_time = time.time()
    train_split, val_split, test_split = datasets
    time_elapsed = time.time() - start_time
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("split initialization time in s for fold {}: {}".format(cur, time_elapsed))
    print("Training on {} samples".format(len(train_split)))
    try: 
        print("Validating on {} samples".format(len(val_split)))
    except TypeError as error:
        print(error)
        print("No validation during training!")
    try: 
        print("Testing on {} samples".format(len(test_split)))
    except TypeError as error:
        print(error)
        print("No test during training!")  

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'mse':
        loss_fn = nn.MSELoss()
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        raise NotImplementedError('The given loss function not implemented!')
    print('Done!')
    
    print('\nInit Model...', end=' ')
    start_time = time.time()
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None:
        model_dict.update({"size_arg": args.model_size})
        
    if args.model_activation is not None:
        model_dict.update({"activation": args.model_activation})
    
    if args.model_type == 'clam_mb_reg':
        model = CLAM_MB_reg(**model_dict)
    else:
        raise NotImplementedError
    
    model.relocate()
    
    print('Done!')
    time_elapsed = time.time() - start_time
    print("model initialization time in s for fold {}: {}".format(cur, time_elapsed))
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    start_time = time.time()
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, train_augm = args.train_augm)
    if val_split is not None:
        val_loader = get_split_loader(val_split,  testing = args.testing)
    if test_split is not None:
        test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    time_elapsed = time.time() - start_time
    print("loader initialization time in s for fold {}: {}".format(cur, time_elapsed))
    
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    train_start_time = time.time()
    epoch_times = 0.

    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, args.patch_pred, writer, loss_fn)

        epoch_elapsed = time.time() - epoch_start_time
        print("training time in s for epoch {}: {}".format(epoch, epoch_elapsed))
        epoch_times += epoch_elapsed

        if val_split is not None:
            stop, best_epoch = validate(cur, epoch, model, val_loader, args.n_classes, args.patch_pred,
                early_stopping, writer, loss_fn, args.results_dir)
        else:
            stop = False
        
        if stop: 
            break

    print("\nTrained {} epoches for fold {}!".format(epoch+1, cur))
        
    train_time_elapsed = time.time() - train_start_time
    
    epoch_times /= epoch
    print("average time in s per epoch: {}\n".format(epoch_times))

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
        print("\nThe best model for fold {} achieved at epoch {}!".format(cur, best_epoch))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if val_split is not None:
        _, val_error, val_r2, val_r2s, val_pearsons, val_ppvals, _= summary(model, val_loader, args.n_classes, args.patch_pred)
        val_pearson = sum(val_pearsons)/len(val_pearsons)
        val_ppval = sum(val_ppvals)/len(val_ppvals)
        print('Val error: {:.4f}, r2: {:.4f}, mean pearson: {:.4f}, mean ppval: {:.4f}'.format(val_error, val_r2, val_pearson, val_ppval))
    else:
        val_error = 2
        val_r2 = np.nan
        val_pearson = np.nan
        val_ppval = -1

    if test_split is not None:
        results_dict, test_error, test_r2, test_r2s, test_pearsons, test_ppvals, acc_logger = summary(model, test_loader, args.n_classes, args.patch_pred)
        test_pearson = sum(test_pearsons)/len(test_pearsons)
        test_ppval = sum(test_ppvals)/len(test_ppvals)
        print('Test error: {:.4f}, r2: {:.4f}, mean pearson: {:.4f}, mean ppval: {:.4f}'.format(test_error, test_r2, test_pearson, test_ppval))
        for i in range(args.n_classes):
            r2, pearson, ppval, count = acc_logger.get_summary_reg(i)
            print('class {}: r2 {}, pearson {}, ppval {}, count {}'.format(i, r2, pearson, ppval, count))
    
            if writer:
                writer.add_scalar('final/test_class_{}_r2'.format(i), r2, 0)
                writer.add_scalar('final/test_class_{}_pearson'.format(i), pearson, 0)
                writer.add_scalar('final/test_class_{}_ppval'.format(i), ppval, 0)
    else:
        test_error = 2
        test_r2 = np.nan
        test_pearson = np.nan
        test_ppval = -1
        results_dict = None

    

    if writer:
        if val_split is not None:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_r2', val_r2, 0)
            writer.add_scalar('final/val_pearson', val_pearson, 0)
            writer.add_scalar('final/val_ppval', val_ppval, 0)
        if test_split is not None:
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_r2', test_r2, 0)
            writer.add_scalar('final/test_pearson', test_pearson, 0)
            writer.add_scalar('final/test_ppval', test_ppval, 0)
    
    writer.close()
    return results_dict, test_r2, val_r2, test_pearson, val_pearson, test_ppval, val_ppval, 1-test_error, 1-val_error, train_time_elapsed

def train_loop(epoch, model, loader, optimizer, n_classes, p_pred, writer = None, loss_fn = None): 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes, reg=True)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)

        if p_pred:
            _, A_raw, results_dict = model(data, patch_pred=p_pred) # A_raw: (1, N) for sb, (k, N) for mb
            patch_pred = results_dict['patch_pred'] # (N, k) for mb
            patch_pred = torch.transpose(patch_pred, 1, 0) # (k, N)
            logits = patch_pred * F.softmax(A_raw, dim=1) # (k, N)
            logits = torch.transpose(torch.sum(logits, dim=1, keepdim=True), 1, 0) # (k, 1) -> (1, k)
        else:
            logits, _, _ = model(data)

        acc_logger.log_reg(logits, targets)
        loss = loss_fn(logits, targets)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, targets: {}, bag_size: {}'.format(batch_idx, loss_value, targets, data.size(0)))
           
        error = symmetric_mean_absolute_percentage_error(logits, targets)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        r2, pearson, ppval, count = acc_logger.get_summary_reg(i)
        print('class {}: r2 {}, pearson {}, ppval {}, count {}'.format(i, r2, pearson, ppval, count))
        if writer:
            writer.add_scalar('train/class_{}_r2'.format(i), r2, epoch)
            writer.add_scalar('train/class_{}_pearson'.format(i), pearson, epoch)
            writer.add_scalar('train/class_{}_ppval'.format(i), ppval, epoch)
        wandb.log({f"train_class_{i}_r2": r2, f"train_class_{i}_pearson": pearson,
                   f"train_class_{i}_ppval": ppval}, step=epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
    wandb.log({"train_loss": train_loss, "train_error": train_error}, step=epoch)

def validate(cur, epoch, model, loader, n_classes, p_pred, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes, reg=True)
    val_loss = 0.
    val_error = 0.

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if p_pred:
                _, A_raw, results_dict = model(data, patch_pred=p_pred) # A_raw: (1, N) for sb, (k, N) for mb
                patch_pred = results_dict['patch_pred'] # (N, k) for mb
                patch_pred = torch.transpose(patch_pred, 1, 0) # (k, N)
                logits = patch_pred * F.softmax(A_raw, dim=1) # (k, N)
                logits = torch.transpose(torch.sum(logits, dim=1, keepdim=True), 1, 0) # (k, 1) -> (1, k)
            else:
                logits, _, _ = model(data)

            acc_logger.log_reg(logits, targets)
            
            loss = loss_fn(logits, targets)
            
            val_loss += loss.item()
            error = symmetric_mean_absolute_percentage_error(logits, targets)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}'.format(val_loss, val_error))
    wandb.log({"val_loss": val_loss, "val_error": val_error}, step=epoch)
    for i in range(n_classes):
        r2, pearson, ppval, count = acc_logger.get_summary_reg(i)
        print('class {}: r2 {}, pearson {}, ppval{}, count {}'.format(i, r2, pearson, ppval, count))
        if writer:
            writer.add_scalar('val/class_{}_r2'.format(i), r2, epoch)
            writer.add_scalar('val/class_{}_pearson'.format(i), pearson, epoch)
            writer.add_scalar('val/class_{}_ppval'.format(i), ppval, epoch)
        wandb.log({f"val_class_{i}_r2": r2, f"val_class_{i}_pearson": pearson,
                   f"val_class_{i}_ppval": ppval}, step=epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, early_stopping.best_epoch	

    return False, -1

def summary(model, loader, n_classes, p_pred):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes, reg=True)
    model.eval()

    test_error = 0.
    r2s = []
    pearsons = []
    ppvals = []

    all_preds = np.zeros((len(loader), n_classes), dtype=np.float64)
    all_targets = np.zeros((len(loader), n_classes), dtype=np.float64)

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if p_pred:
                _, A_raw, results_dict = model(data, patch_pred=p_pred) # A_raw: (1, N) for sb, (k, N) for mb
                patch_pred = results_dict['patch_pred'] # (N, k) for mb
                patch_pred = torch.transpose(patch_pred, 1, 0) # (k, N)
                logits = patch_pred * F.softmax(A_raw, dim=1) # (k, N)
                logits = torch.transpose(torch.sum(logits, dim=1, keepdim=True), 1, 0) # (k, 1) -> (1, k)
            else:
                logits, _, _ = model(data)

        acc_logger.log_reg(logits, targets)
        all_preds[batch_idx] = logits.cpu().numpy()
        all_targets[batch_idx] = targets.cpu().numpy()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'pred': logits, 'targets': targets.cpu().numpy()}})
        error = symmetric_mean_absolute_percentage_error(logits, targets)
        test_error += error.cpu().numpy()

    test_error /= len(loader)

    r2 = r2_score(torch.FloatTensor(all_preds), torch.FloatTensor(all_targets))
    
    for c in range(n_classes):
        r2s.append(r2_score(torch.FloatTensor(all_preds[:, c]), torch.FloatTensor(all_targets[:, c])))
        pearson, ppval = pearsonr(all_preds[:, c], all_targets[:, c])
        pearsons.append(pearson)
        ppvals.append(ppval)

    return patient_results, test_error, r2, r2s, pearsons, ppvals, acc_logger
