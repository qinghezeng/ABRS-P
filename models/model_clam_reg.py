#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:23:45 2021

@author: Q Zeng
"""

from models.model_clam import CLAM_MB, Attn_Net_Gated, Attn_Net
from utils.utils import initialize_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

class CLAM_MB_reg(CLAM_MB):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, activation=None):
        super(CLAM_MB_reg, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "small-768": [768, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        self.activation = activation
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.n_classes = n_classes
        initialize_weights(self)
        
    def forward(self, h, label=None, return_features=False, attention_only=False, patch_pred=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h) 
        logits = torch.empty(1, self.n_classes).float().to(device)
        plogits = torch.empty(h.size()[0], self.n_classes).float().to(device) # Nxk
        for c in range(self.n_classes):
            tmp = self.classifiers[c](M[c])
            ptmp = torch.squeeze(self.classifiers[c](h), 1) # Nx512 -> Nx1 -> N
            if not self.activation:
                logits[0, c] = tmp
                plogits[:, c] = ptmp
            else:
                if self.activation == 'softplus':
                    logits[0, c] = F.softplus(tmp)
                    plogits[:, c] = F.softplus(ptmp)
                else:
                    raise NotImplementedError()

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        if patch_pred:	
            results_dict.update({'patch_pred': plogits}) # plogits
            
        return logits, A_raw, results_dict



