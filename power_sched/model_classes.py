#/usr/bin/env python3

import numpy as np
import scipy.stats as st
import operator
from functools import reduce

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.cuda

import qpth
from qpth.qp import QPFunction

import ipdb

class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()

        # Initialize linear layer with least squares solution
        X_ = np.hstack([X, np.ones((X.shape[0],1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))
        
        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W,b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1,:].T)
        b.data = torch.Tensor(Theta[-1,:])
        
        # Set up non-linear network of 
        # Linear -> BatchNorm -> ReLU -> Dropout layers
        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add, 
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)] 
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        self.sig = Parameter(torch.ones(1, Y.shape[1]).cuda())
        
    def forward(self, x):
        return self.lin(x) + self.net(x), \
            self.sig.expand(x.size(0), self.sig.size(1))
    
    def set_sig(self, X, Y):
        Y_pred = self.lin(X) + self.net(X)
        var = torch.mean((Y_pred-Y)**2, 0)
        self.sig.data = torch.sqrt(var).cuda().data.unsqueeze(0)


class GLinearApprox(Function):
    """ Linear (gradient) approximation of G function at z"""
    def __init__(self, gamma_under, gamma_over):
        self.gamma_under = gamma_under
        self.gamma_over = gamma_over
    
    def forward(self, z, mu, sig):
        self.save_for_backward(z, mu, sig)
        p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
        return torch.DoubleTensor((self.gamma_under + self.gamma_over) * p.cdf(
            z.cpu().numpy()) - self.gamma_under).cuda()
    
    def backward(self, grad_output):
        z, mu, sig = self.saved_tensors
        p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
        pz = torch.DoubleTensor(p.pdf(z.cpu().numpy())).cuda()
        
        dz = (self.gamma_under + self.gamma_over) * pz
        dmu = -dz
        dsig = -(self.gamma_under + self.gamma_over)*(z-mu) / sig * pz
        return grad_output * dz, grad_output * dmu, grad_output * dsig


class GQuadraticApprox(Function):
    """ Quadratic (gradient) approximation of G function at z"""
    def __init__(self, gamma_under, gamma_over):
        self.gamma_under = gamma_under
        self.gamma_over = gamma_over
    
    def forward(self, z, mu, sig):
        self.save_for_backward(z, mu, sig)
        p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
        return torch.DoubleTensor((self.gamma_under + self.gamma_over) * p.pdf(
            z.cpu().numpy())).cuda()
    
    def backward(self, grad_output):
        z, mu, sig = self.saved_tensors
        p = st.norm(mu.cpu().numpy(),sig.cpu().numpy())
        pz = torch.DoubleTensor(p.pdf(z.cpu().numpy())).cuda()
        
        dz = -(self.gamma_under + self.gamma_over) * (z-mu) / (sig**2) * pz
        dmu = -dz
        dsig = (self.gamma_under + self.gamma_over) * ((z-mu)**2 - sig**2) / \
            (sig**3) * pz
        
        return grad_output * dz, grad_output * dmu, grad_output * dsig


class SolveSchedulingQP(nn.Module):
    """ Solve a single SQP iteration of the scheduling problem"""
    def __init__(self, params):
        super(SolveSchedulingQP, self).__init__()
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = Variable(torch.DoubleTensor(np.vstack([D,-D])).cuda())
        self.h = Variable((self.c_ramp * torch.ones((self.n - 1) * 2))\
            .double().cuda())
        self.e = Variable(torch.Tensor().double().cuda())
        
    def forward(self, z0, mu, dg, d2g):
        nBatch, n = z0.size()
        
        Q = torch.cat([torch.diag(d2g[i] + 1).unsqueeze(0) 
            for i in range(nBatch)], 0).double()
        p = (dg - d2g*z0 - mu).double()
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        
        out = QPFunction(verbose=False)(Q, p, G, h, self.e, self.e)
        return out


class SolveScheduling(nn.Module):
    """ Solve the entire scheduling problem, using sequential quadratic 
        programming. """
    def __init__(self, params):
        super(SolveScheduling, self).__init__()
        self.params = params
        self.c_ramp = params["c_ramp"]
        self.n = params["n"]
        
        D = np.eye(self.n - 1, self.n) - np.eye(self.n - 1, self.n, 1)
        self.G = Variable(torch.DoubleTensor(np.vstack([D, -D])).cuda())
        self.h = Variable((self.c_ramp * torch.ones((self.n - 1) * 2))\
            .double().cuda())
        self.e = Variable(torch.Tensor().double().cuda())
        
    def forward(self, mu, sig):
        nBatch, n = mu.size()
        
        # Find the solution via sequential quadratic programming, 
        # not preserving gradients
        z0 = Variable(1. * mu.data, requires_grad=False)
        mu0 = Variable(1. * mu.data, requires_grad=False)
        sig0 = Variable(1. * sig.data, requires_grad=False)
        for i in range(20):
            dg = GLinearApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            d2g = GQuadraticApprox(self.params["gamma_under"], 
                self.params["gamma_over"])(z0, mu0, sig0)
            z0_new = SolveSchedulingQP(self.params)(z0, mu0, dg, d2g)
            solution_diff = (z0-z0_new).norm().data[0]
            print("+ SQP Iter: {}, Solution diff = {}".format(i, solution_diff))
            z0 = z0_new
            if solution_diff < 1e-10:
                break
                  
        # Now that we found the solution, compute the gradient-propagating 
        # version at the solution
        dg = GLinearApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        d2g = GQuadraticApprox(self.params["gamma_under"], 
            self.params["gamma_over"])(z0, mu, sig)
        return SolveSchedulingQP(self.params)(z0, mu, dg, d2g)
