#!/usr/bin/env python3

import argparse

import cvxpy as cp
import numpy as np

import os

from datetime import datetime

try: import setGPU
except ImportError: pass

import torch
from torch.autograd import Variable

import setproctitle

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import mle, policy_net, task_net, plot


def main():
    parser = argparse.ArgumentParser(
        description='Run newsvendor task net experiments.')
    parser.add_argument('--save', type=str, required=True, 
        metavar='save-folder', help='save folder path')
    parser.add_argument('--nRuns', type=int, default=10,
        metavar='runs', help='number of runs')
    parser.add_argument('--trueModel', type=str, 
        choices=['linear', 'nonlinear'], default='linear', 
        help='true y|x distribution')
    args = parser.parse_args()

    setproctitle.setproctitle('pdonti.' + args.save + args.trueModel)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    results_file = os.path.join(args.save, 
        'inventory_results_{}.csv'.format(args.trueModel))

    # Cost params for newsvendor task loss
    params = init_newsvendor_params()

    # Randomly generate true params for p(y|x;θ).
    # Set with_seed=True to replicate paper true params.
    Theta_true_lin, Theta_true_sq = init_theta_true(
        params, is_linear=(args.trueModel == 'linear'), with_seed=True)

    # Test data. Set with_seed=True to replicate paper test data.
    X_test, Y_test = gen_data(1000, params, Theta_true_lin, Theta_true_sq,
                              with_seed=True)

    # MLE with true params
    f_eval_mle_t, z_buy_t, f_opt_t = mle.newsvendor_eval(
        X_test, Y_test, Theta_true_lin, Theta_true_sq, params)
    print(np.mean(f_eval_mle_t))
    mle_true_score = np.mean(f_eval_mle_t)

    with open(results_file, 'w') as f:
        f.write('{},{}\n'.format('mle_true:', mle_true_score))
        f.write('{},{},{},{},{}\n'.format(
            'm', 'mle', 'policy', 'task-linear', 'task-nonlinear'))

    for run in range(args.nRuns):
        for m in [100, 200, 300, 500, 1000, 3000, 5000, 10000]:

            with open(results_file, 'a') as f:
                f.write('\n{},'.format(m))

            # Generate data based on true params
            try:
                X, Y = gen_data(m, params, Theta_true_lin, Theta_true_sq)
            except Exception as e:
                log_error_and_write(e, args, m, run, 'gen', 
                    results_file, newline=True)


            # MLE with linear softmax regression
            try:
                Theta_est = mle.linear_softmax_reg(X, Y, params)
                f_eval_mle, z_buy, f_opt = \
                    mle.newsvendor_eval(X_test, Y_test, Theta_est, 
                        np.zeros((params['n'], len(params['d']))),
                        params)
                mle_score = np.mean(f_eval_mle)

                print(mle_score)
                with open(results_file, 'a') as f:
                    f.write('{},'.format(mle_score))
            except Exception as e:
                log_error_and_write(e, args, m, run, 'mle', results_file)

            # Pure end-to-end policy neural net
            try:
                policy_score = policy_net.run_policy_net(
                    X, Y, X_test, Y_test, params)

                print(policy_score)
                with open(results_file, 'a') as f:
                    f.write('{},'.format(policy_score))
            except Exception as e:
                log_error_and_write(e, args, m, run, 'policy', results_file)


            # Model-based end-to-end model (linear)
            try:
                e2e_lin_score = task_net.run_task_net(
                    X, Y, X_test, Y_test, params)

                print(e2e_lin_score)
                with open(results_file, 'a') as f:
                    f.write('{},'.format(e2e_lin_score))
            except Exception as e:
                log_error_and_write(e, args, m, run, 
                    'task-linear', results_file)

            # Model-based end-to-end model (nonlinear)
            try:
                e2e_nonlin_score = task_net.run_task_net(
                    X, Y, X_test, Y_test, params, is_nonlinear=True)

                print(e2e_nonlin_score)
                with open(results_file, 'a') as f:
                    f.write('{}\n'.format(e2e_nonlin_score))
            except Exception as e:
                log_error_and_write(e, args, m, run, results_file, 
                    'task-nonlinear', newline=True)

            # Plot results as we go
            try:
                plot.plot_results(args.save, args.trueModel)
            except Exception as e:
                with open(os.path.join(args.save, 
                    'errors-{}.log'.format(args.trueModel)), 'a') as f:
                    f.write('{}: m {}, model {}, run {}: {}\n'.format(
                        datetime.now(), m, 'plot', run, e))

    # Plot results
    plot.plot_results(args.save, args.trueModel)
    


def init_newsvendor_params():
    params = {}

    # Ordering costs
    params['c_lin'] = 10
    params['c_quad'] = 2.0

    # Over-order penalties
    params['b_lin'] = 30
    params['b_quad'] = 14

    # Under-order penalties
    params['h_lin'] = 10
    params['h_quad'] = 2

    # Discrete demands
    params['d'] = np.array([1, 2, 5, 10, 20]).astype(np.float32)

    # Number of features
    params['n'] = 20

    return params


def init_theta_true(params, is_linear=True, with_seed=False):
    if is_linear:
        # Linear true model (py ∝ exp(θX))
        np.random.seed(42) if with_seed else np.random.seed(None)
        Theta_true_lin = np.random.randn(params['n'], len(params['d']))
        Theta_true_sq = np.zeros((params['n'], len(params['d'])))
    else:
        # Squared true model (py ∝ exp((θX)^2))
        Theta_true_lin = np.zeros((params['n'], len(params['d'])))
        np.random.seed(42) if with_seed else np.random.seed(None)
        Theta_true_sq = np.random.randn(params['n'], len(params['d']))

    np.random.seed(None)

    return Theta_true_lin, Theta_true_sq


def gen_data(m, params, Theta_true_lin, Theta_true_sq, with_seed=False):
    np.random.seed(0) if with_seed else np.random.seed(None)
    X  = np.random.randn(m, params['n'])

    PY = np.exp(X.dot(Theta_true_lin) + (X.dot(Theta_true_sq)) ** 2)
    PY = PY / np.sum(PY, axis=1)[:, None]

    # Generate demand realizations
    Y  = np.where(np.cumsum(np.random.rand(m)[:, None]
                < np.cumsum(PY, axis=1), axis=1) == 1)[1]
    Y  = np.eye(len(params['d']))[Y, :]

    np.random.seed(None)

    return X, Y

def log_error_and_write(e, args, m, run, model, results_file, newline=False):
    with open(os.path.join(args.save, 
        'errors-{}.log'.format(args.trueModel)), 'a') as f:
        f.write('{}: m {}, model {}, run {}: {}\n'.format(
            datetime.now(), m, model, run, e))
    with open(results_file, 'a') as f:
        f.write('\n' if newline else ',')


if __name__=='__main__':
    main()
