#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import operator
from functools import reduce

import batch
from constants import *

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

def run_policy_net(X_train, Y_train, X_test, Y_test, params, is_nonlinear=False):

    if is_nonlinear:
        # Non-linear model, use ADAM step size 1e-3
        layer_sizes = [params['n'], 200, 200, 1]
        layers = reduce(operator.add, [[nn.Linear(a,b), nn.BatchNorm1d(b), 
                                        nn.ReLU(), nn.Dropout(p=0.2)]   # TODO: Why is this 0.2? (others are 0.5)
                          for a,b in zip(layer_sizes[0:-2], layer_sizes[1:-1])])
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        model = nn.Sequential(*layers)
        step_size = 1e-3
    else:
        # Linear model, use ADAM step size 1e-2
        model = nn.Sequential(
            nn.Linear(params['n'], 1)
        )
        step_size = 1e-2

    if USE_GPU:
        model = model.cuda()

    X_train_t = torch.tensor(X_train, dtype=torch.float, device=DEVICE)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
    X_test_t  = torch.tensor(X_test, dtype=torch.float, device=DEVICE)
    Y_test_t  = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)
    d_ = torch.tensor(params['d'], dtype=torch.float, device=DEVICE)

    # Expected inventory cost
    cost = lambda Z, Y : (params['c_lin'] * Z + 0.5 * params['c_quad'] * (Z**2) +
                      params['b_lin'] * (Y.mv(d_).view(-1,1)-Z).clamp(min=0) +
                      0.5 * params['b_quad'] * (Y.mv(d_).view(-1,1)-Z).clamp(min=0)**2 +
                      params['h_lin'] * (Z-Y.mv(d_).view(-1,1)).clamp(min=0) +
                      0.5 * params['h_quad'] * (Z-Y.mv(d_).view(-1,1)).clamp(min=0)**2) \
                    .mean()

    opt = optim.Adam(model.parameters(), lr=step_size)

    for i in range(1000):

        model.eval()
        test_cost = batch.get_cost(100, i, model, X_test_t, Y_test_t, cost)

        model.train()
        train_cost = batch_train(150, i, X_train_t, Y_train_t, model, opt, cost)

        print(train_cost.item(), test_cost.item())

    return test_cost.item()


def batch_train(batch_sz, epoch, X_train_t, Y_train_t, model, opt, cost_fn):
    train_cost = 0
    batch_data_, batch_targets_ = \
        batch.get_vars(batch_sz, X_train_t, Y_train_t)
    size = batch_sz

    for i in range(0, X_train_t.size(0), batch_sz):

        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_train_t.size(0):
            size = X_train_t.size(0) - i
            batch_data_, batch_targets_ = \
                batch.get_vars(size, X_train_t, Y_train_t)
        
        batch_data_.data[:] = X_train_t[i:i+size]
        batch_targets_.data[:] = Y_train_t[i:i+size]

        opt.zero_grad()
        preds = model(batch_data_)
        batch_cost = cost_fn(preds, batch_targets_)
        batch_cost.backward()
        opt.step()

        ## Keep running average of loss
        train_cost += (batch_cost - train_cost) * size / (i + size)

        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
            epoch, i+size, X_train_t.size(0),
            float(i+size)/X_train_t.size(0)*100,
            batch_cost.item()))

    return train_cost
