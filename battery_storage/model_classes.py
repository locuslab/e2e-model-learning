#/usr/bin/env python3

import operator
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torch.cuda

import qpth
from qpth.qp import QPFunction

from block import block

class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes, T):
        super(Net, self).__init__()
        # Initialize linear layer with least squares solution
        X_ = np.hstack([X.cpu().numpy(), np.ones((X.size(0),1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y.cpu().numpy()))
        
        self.lin = nn.Linear(X.size(1), Y.size(1))
        W,b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1,:].T)
        b.data = torch.Tensor(Theta[-1,:])


        # Set up non-linear network of
        # Linear -> BatchNorm -> ReLU -> Dropout layers
        layer_sizes = [X.size(1)] + hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.size(1))]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.lin(x) + self.net(x)

class ScheduleBattery(nn.Module):
    ''' Get battery schedule that maximizes objective '''
    
    def __init__(self, params):
        super(ScheduleBattery, self).__init__()
        self.T = params['T']
        T = params['T']
        eps = params['epsilon']
        IT = torch.eye(T)
        eff = params['eff']
        in_max = params['in_max']
        out_max = params['out_max']
        self.B = params['B']
        self.lam = params['lambda']

        D1 = torch.cat([torch.eye(T-1), torch.zeros(1, T-1)], 0)
        D2 = torch.cat([torch.zeros(1, T-1), torch.eye(T-1)], 0)
        
        self.Q = Variable(block([[eps * torch.eye(T), 0, 0],
                  [0, eps * torch.eye(T), 0],
                  [0, 0, self.lam * torch.eye(T)]])).cuda()
        
        Ae_list = [[torch.zeros(1,T), torch.zeros(1,T), torch.ones(1,1), torch.zeros(1,T-1)],
                   [ D1.t() * eff, -D1.t(), D1.t()-D2.t()]]
        self.Ae = Variable(torch.cat(map(lambda x: torch.cat(x, 1), Ae_list), 0)).cuda()
        self.be = Variable(torch.cat([(self.B/2) * torch.ones(1), torch.zeros(T-1)])).cuda()

        self.A = Variable(
            block([[torch.eye(T), 0, 0],
               [-torch.eye(T), 0, 0],
               [0, torch.eye(T), 0],
               [0, -torch.eye(T), 0],
               [0, 0, torch.eye(T)],
               [0, 0, -torch.eye(T)]])).cuda()
        self.b = Variable(torch.Tensor(
            [in_max]*T + [0]*T + [out_max]*T + [0]*T + [self.B]*T + [0]*T)).cuda()

        
    def forward(self, log_prices):
        prices = torch.exp(log_prices)
        
        nBatch = prices.size(0)
        T = self.T

        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        c = torch.cat(
            [prices, -prices, 
            -Variable(self.lam * self.B * torch.ones(T)).unsqueeze(0).expand(nBatch,T).cuda()], 
            1)
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))
        Ae = self.Ae.unsqueeze(0).expand(nBatch, self.Ae.size(0), self.Ae.size(1))
        be = self.be.unsqueeze(0).expand(nBatch, self.be.size(0))
                
        out = QPFunction(verbose=True)\
            (Q.double(), c.double(), A.double(), b.double(), Ae.double(), be.double())
        
        return out

