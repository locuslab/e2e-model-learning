import numpy as np
import operator
from functools import reduce

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torch.cuda

import importlib
import qpth
from qpth.qp import QPFunction

from block import block

def get_vars(batch_sz, X_test_t, Y_test_t):
    batch_data_ = Variable(torch.Tensor(batch_sz, X_test_t.size(1)), 
        requires_grad=False).cuda()
    batch_targets_ = Variable(torch.Tensor(batch_sz, Y_test_t.size(1)), 
        requires_grad=False).cuda()
    return batch_data_, batch_targets_

def get_vars_scalar_out(batch_sz, X_test_t, Y_test_t):
    batch_data_ = Variable(torch.Tensor(batch_sz, X_test_t.size(1)), 
        requires_grad=False).cuda()
    batch_targets_ = Variable(torch.LongTensor(batch_sz),
        requires_grad=False).cuda()
    return batch_data_, batch_targets_

# General batch evaluation
def get_cost_helper(batch_sz, epoch, model, X_test_t, Y_test_t, 
    loss_fn, var_getter_fn):
    test_cost = 0

    batch_data_, batch_targets_ = var_getter_fn(
        batch_sz, X_test_t, Y_test_t)
    size = batch_sz

    for i in range(0, X_test_t.size(0), batch_sz):

        # Deal with potentially incomplete (last) batch
        if i + batch_sz  > X_test_t.size(0):
            size = X_test_t.size(0) - i
            batch_data_, batch_targets_ = var_getter_fn(
                size, X_test_t, Y_test_t)
        
        batch_data_.data[:] = X_test_t[i:i+size]
        batch_targets_.data[:] = Y_test_t[i:i+size]

        preds = model(batch_data_)
        batch_cost = loss_fn(preds, batch_targets_)

        # Keep running average of loss
        test_cost += (batch_cost - test_cost) * size / (i + size)

    print('TEST SET RESULTS:' + ' ' * 20)
    print('Average loss: {:.4f}'.format(test_cost.data[0]))

    return test_cost

def get_cost(batch_sz, epoch, model, X_test_t, Y_test_t, loss_fn):
    return get_cost_helper(batch_sz, epoch, model, X_test_t, Y_test_t, 
        loss_fn, get_vars)

def get_cost_nll(batch_sz, epoch, model, X_test_t, Y_test_t, loss_fn):
    return get_cost_helper(batch_sz, epoch, model, X_test_t, Y_test_t, 
        loss_fn, get_vars_scalar_out)

