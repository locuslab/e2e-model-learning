#!/usr/bin/env python3

import numpy as np
import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim

from qpth.qp import QPFunction

import batch
from constants import *

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)


class SolveNewsvendor(nn.Module):
    """ Solve newsvendor scheduling problem """
    def __init__(self, params, eps=1e-2):
        super(SolveNewsvendor, self).__init__()
        k = len(params['d'])
        self.Q = torch.diag(torch.Tensor(
            [params['c_quad']] + [params['b_quad']]*k + [params['h_quad']]*k))
        self.p = torch.Tensor(
            [params['c_lin']] + [params['b_lin']]*k + [params['h_lin']]*k)
        self.G = torch.cat([
            torch.cat([-torch.ones(k,1), -torch.eye(k), torch.zeros(k,k)], 1),
            torch.cat([torch.ones(k,1), torch.zeros(k,k), -torch.eye(k)], 1),
            -torch.eye(1 + 2*k)], 0)
        self.h = torch.Tensor(
            np.concatenate([-params['d'], params['d'], np.zeros(1+ 2*k)]))
        self.one = torch.Tensor([1])
        self.eps_eye = eps * torch.eye(1 + 2*k).unsqueeze(0)

        if USE_GPU:
            self.Q = self.Q.cuda()
            self.p = self.p.cuda()
            self.G = self.G.cuda()
            self.h = self.h.cuda()
            self.one = self.one.cuda()
            self.eps_eye = self.eps_eye.cuda()

    def forward(self, y):
        nBatch, k = y.size()

        Q_scale = torch.cat([torch.diag(torch.cat(
            [self.one, y[i], y[i]])).unsqueeze(0) for i in range(nBatch)], 0)
        Q = self.Q.unsqueeze(0).expand_as(Q_scale).mul(Q_scale)
        p_scale = torch.cat([torch.ones(nBatch, 1, device=DEVICE), y, y], 1)
        p = self.p.unsqueeze(0).expand_as(p_scale).mul(p_scale)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        e = torch.DoubleTensor()
        if USE_GPU:
            e = e.cuda()

        out = QPFunction(verbose=False)\
            (Q.double(), p.double(), G.double(), h.double(), e, e).float()

        return out[:,:1]


def run_task_net(X, Y, X_test, Y_test, params, is_nonlinear=False):

    # Training/validation split
    th_frac = 0.8
    inds = np.random.permutation(X.shape[0])
    train_inds = inds[:int(X.shape[0]*th_frac)]
    hold_inds =  inds[int(X.shape[0]*th_frac):]
    X_train, X_hold = X[train_inds, :], X[hold_inds, :]
    Y_train, Y_hold = Y[train_inds, :], Y[hold_inds, :]

    X_train_t = torch.tensor(X_train, dtype=torch.float, device=DEVICE)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
    X_hold_t = torch.tensor(X_hold, dtype=torch.float, device=DEVICE)
    Y_hold_t = torch.tensor(Y_hold, dtype=torch.float, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float, device=DEVICE)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)

    Y_train_int_t = torch.where(Y_train_t)[1].detach()
    Y_hold_int_t = torch.where(Y_hold_t)[1].detach()
    Y_test_int_t = torch.where(Y_test_t)[1].detach()

    d_ = torch.tensor(params['d'], dtype=torch.float, device=DEVICE)

    # Expected inventory cost and solver for newsvendor scheduling problem
    cost = lambda Z, Y : (params['c_lin'] * Z + 0.5 * params['c_quad'] * (Z**2) +
                          params['b_lin'] * (Y.mv(d_).view(-1,1)-Z).clamp(min=0) +
                          0.5 * params['b_quad'] * (Y.mv(d_).view(-1,1)-Z).clamp(min=0)**2 +
                          params['h_lin'] * (Z-Y.mv(d_).view(-1,1)).clamp(min=0) +
                          0.5 * params['h_quad'] * (Z-Y.mv(d_).view(-1,1)).clamp(min=0)**2) \
                        .mean()
    newsvendor_solve = SolveNewsvendor(params)
    if USE_GPU:
        newsvendor_solve = newsvendor_solve.cuda()
    cost_news_fn = lambda x, y: cost(newsvendor_solve(x), y)

    nll = nn.NLLLoss()
    if USE_GPU:
        nll = nll.cuda()
    lam = 10.0  # regularization

    if is_nonlinear:
        # Non-linear model, use ADAM step size 1e-3
        layer_sizes = [X_train.shape[1], 200, 200, Y_train.shape[1]]
        layers = reduce(operator.add, [[nn.Linear(a,b), nn.BatchNorm1d(b), 
                                        nn.ReLU(), nn.Dropout(p=0.5)]
                          for a,b in zip(layer_sizes[0:-2], layer_sizes[1:-1])])
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax()]
        model = nn.Sequential(*layers)
        step_size = 1e-3
    else:
        # Linear model, use ADAM step size 1e-2
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], Y_train.shape[1]),
            nn.Softmax()
        )
        step_size = 1e-2

    if USE_GPU:
        model = model.cuda()

    opt = optim.Adam(model.parameters(), lr=step_size)

    # For early stopping
    hold_costs, test_costs = [], []
    num_stop_rounds = 20

    for i in range(1000):
        model.eval()
        test_cost = batch.get_cost(
            100, i, model, X_test_t, Y_test_t, cost_news_fn)
        test_nll = batch.get_cost_nll(
            100, i, model, X_test_t, Y_test_int_t, nll)

        hold_cost = batch.get_cost(
            100, i, model, X_hold_t, Y_hold_t, cost_news_fn)
        hold_nll  = batch.get_cost_nll(
            100, i, model, X_hold_t, Y_hold_int_t, nll)

        model.train()
        train_cost, train_nll = batch_train(150, i, X_train_t, Y_train_t, 
            Y_train_int_t, model, cost_news_fn, nll, opt, lam)

        print(i, train_cost.item(), train_nll.item(), test_cost.item(), 
              test_nll.item(), hold_cost.item(), hold_nll.item())

        # Early stopping
        test_costs.append(test_cost.item())
        hold_costs.append(hold_cost.item())
        if i > 0 and i % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            # Stop if current cost is worst in num_stop_rounds rounds
            if max(hold_costs) == hold_cost.item():
                print(test_costs[idx])
                return(test_costs[idx])
            else:
                # Keep only "best" round
                hold_costs = [hold_costs[idx]]
                test_costs = [test_costs[idx]]

    # In case of no early stopping, return best run so far
    idx = hold_costs.index(min(hold_costs))
    return test_costs[idx]

def batch_train(batch_sz, epoch, X_train_t, Y_train_t, Y_train_int_t,
    model, cost_fn_news, nll, opt, lam):

    train_cost_agg = 0
    train_nll_agg = 0

    batch_data_, batch_targets_ = \
        batch.get_vars(batch_sz, X_train_t, Y_train_t)
    _, batch_targets_int_ = \
        batch.get_vars_scalar_out(batch_sz, X_train_t, Y_train_int_t)
    size = batch_sz

    for i in range(0, X_train_t.size(0), batch_sz):

        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_train_t.size(0):
            size = X_train_t.size(0) - i
            batch_data_, batch_targets_ = batch.get_vars(
                size, X_train_t, Y_train_t)
            _, batch_targets_int_ = batch.get_vars_scalar_out(
                size, X_train_t, Y_train_int_t)

        batch_data_.data[:] = X_train_t[i:i+size]
        batch_targets_.data[:] = Y_train_t[i:i+size]
        batch_targets_int_.data[:] = Y_train_int_t[i:i+size]

        opt.zero_grad()
        preds = model(batch_data_)
        train_cost = cost_fn_news(preds, batch_targets_)
        train_nll  = nll(preds, batch_targets_int_)

        (train_cost + lam * train_nll).backward()
        opt.step()

        # Keep running average of losses
        train_cost_agg += \
            (train_cost - train_cost_agg) * batch_sz / (i + batch_sz)
        train_nll_agg += \
            (train_nll - train_nll_agg) * batch_sz / (i + batch_sz)

        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
            epoch, i+batch_sz, X_train_t.size(0),
            float(i+batch_sz)/X_train_t.size(0)*100,
            train_cost.item()))

    return train_cost_agg, train_nll_agg
