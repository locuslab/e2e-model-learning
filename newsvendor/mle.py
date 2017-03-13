#!/usr/bin/env python3

import cvxpy as cp
import numpy as np

# Linear softmax regression given X, Y
def linear_softmax_reg(X, Y, params):
    m, n = X.shape[0], X.shape[1]
    Theta = cp.Variable(n, len(params['d']))
    f = cp.sum_entries(cp.log_sum_exp(X*Theta, axis=1) -
                       cp.sum_entries(cp.mul_elemwise(Y, X*Theta), axis=1)) / m
    lam = 1e-5 # regularization
    cp.Problem(cp.Minimize(f + lam * cp.sum_squares(Theta)), []).solve()
    Theta = np.asarray(Theta.value)
    return Theta

# Optimize expected value of inventory allocation 
def newsvendor_opt(params, py):
    z = cp.Variable(1)
    d = params['d']
    f = (params['c_lin'] * z + 0.5 * params['c_quad'] * cp.square(z) +
        py.T * (params['b_lin'] * cp.pos(d-z) + 
                0.5 * params['b_quad'] * cp.square(cp.pos(d-z)) +
                params['h_lin'] * cp.pos(z-d) +
                0.5 * params['h_quad'] * cp.square(cp.pos(z-d)) ))
    fval = cp.Problem(cp.Minimize(f), [z >= 0]).solve()
    return z.value, fval

# Inventory ordering cost given demand realization 
def f_obj(z, d, params):
    return (params['c_lin'] * z + 0.5 * params['c_quad'] * (z**2) + 
            params['b_lin'] * np.maximum(d-z, 0) + 
            0.5 * params['b_quad'] * np.maximum(d-z, 0)**2 + 
            params['h_lin'] * np.maximum(z-d, 0) +
            0.5 * params['h_quad'] * np.maximum(z-d, 0)**2)

# Eval inventory problem performance given parameter estimation
def newsvendor_eval(X, Y, Theta_lin, Theta_sq, params):
    m   = X.shape[0]
    # TODO: deal with overflow
    py  = np.exp(X.dot(Theta_lin) + X.dot(Theta_sq) ** 2)
    py /= np.sum(py, axis=1)[:,None]

    f_eval, f_opt, z_buy = np.zeros(m), np.zeros(m), np.zeros(m)
    for i in range(m):
        z_buy[i], _ = newsvendor_opt(params, py[i])
        f_eval[i] = f_obj(z_buy[i], params['d'].dot(Y[i]), params)
        z_buy_opt, f_opt[i] = newsvendor_opt(params, Y[i])
    return f_eval, z_buy, f_opt
