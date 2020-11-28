#!/usr/bin/env python3

import argparse
import setproctitle

import scipy.io as sio
import numpy as np

import torch

import importlib
try: import setGPU
except ImportError: pass

import os

import model_classes, nets, calc_stats
from constants import *

from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

def main():
    parser = argparse.ArgumentParser(
        description='Run storage task net experiments.')
    parser.add_argument('--save', type=str, 
        metavar='save-folder', help='prefix to add to save path')
    parser.add_argument('--nRuns', type=int, default=10,
        metavar='runs', help='number of runs')
    parser.add_argument('--paramSet', type=int, choices=range(4), default=0,
        metavar='hyperparams', help='(lambda, epsilon) in given row of Table 1')
    args = parser.parse_args()


    save_folder_main = 'params{}'.format(args.paramSet) if args.save is None \
        else '{}-params{}'.format(args.save, args.paramSet)
    save_folder_main = os.path.join('results', save_folder_main)

    setproctitle.setproctitle('storage-{}'.format(args.paramSet))

    # Initialize problem parameters
    params = init_params(args.paramSet)

    bsz = 500

    # Train, test split
    train_frac = 0.8

    input_tensors = get_train_test_split(params, train_frac)
    loaders = get_loaders_tt(input_tensors, bsz)

    if not os.path.exists(save_folder_main):
        os.makedirs(save_folder_main)

    for run in range(args.nRuns):

        save_folder = os.path.join(save_folder_main, str(run))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Randomly construct hold-out set for task net training.
        tensors_task = get_train_hold_split(input_tensors, 0.8, save_folder)
        loaders_task = get_loaders_tth(tensors_task, bsz)

        # Run and eval rmse-minimizing net
        model_rmse = model_classes.Net(
            tensors_task['X_train'], tensors_task['Y_train'], [200, 200], params['T'])
        if USE_GPU:
            model_rmse = model_rmse.cuda()
        model_rmse = nets.run_rmse_net(model_rmse, loaders_task, params, tensors_task)
        nets.eval_net('rmse_net', model_rmse, loaders_task, params, save_folder)

        # Run and eval task-minimizing net
        model_task = model_classes.Net(
            tensors_task['X_train'], tensors_task['Y_train'], [200, 200], params['T'])
        if USE_GPU:
            model_task = model_task.cuda()
        model_task = nets.run_rmse_net(model_task, loaders_task, params, tensors_task)  # seed with rmse soln
        model_task = \
            nets.run_task_net(model_task, loaders_task, params, args, tensors_task)
        nets.eval_net('task_net', model_task, loaders_task, params, save_folder)

    calc_stats.calc_stats(map(
        lambda x: os.path.join(save_folder_main, str(x)), range(args.nRuns)), save_folder_main)


def init_params(param_set):

    # potential (lambda, epsilon) pairs for experiment
    hyperparams = [[0.1, 0.05], [1, 0.5], [10, 5], [35, 15]]

    params = {}

    # Battery capacity
    params['B'] = 1

    # Battery efficiency
    params['eff'] = 0.9

    # Battery max power in
    params['in_max'] = 0.5
    
    # Battery max power out
    params['out_max'] = 0.2
    
    # Number of horizons
    params['T'] = 24

    # Preference for battery staying in middle of range
    params['lambda'] = hyperparams[param_set][0]

    # Regularize z_in and z_out
    params['epsilon'] = hyperparams[param_set][1]

    return params


def get_features_labels(params):
    # TODO predict lmp instead of energy price?
    tz = pytz.timezone('America/New_York')
    df = pd.read_csv('storage_data.csv', parse_dates=[0])
    df['date'] = df['datetime'].apply(lambda x: x.date())
    df['hour'] = df['datetime'].apply(lambda x: x.hour)

    # Prices
    df_prices = df.pivot(index='date', columns='hour', values='da_price')
    df_prices = df_prices.apply(lambda x: pd.to_numeric(x), axis=1)
    df_prices = df_prices.transpose().fillna(method='backfill').transpose()
    df_prices = df_prices.transpose().fillna(method='ffill').transpose()

    # Filtering some outliers
    df_prices_filtered = df_prices.applymap(lambda x: np.nan if x > 500 else x).dropna()
    df_prices_filtered = df_prices_filtered.applymap(lambda x: np.nan if x <= 0 else x).dropna()

    df_prices_log_filtered = np.log(df_prices_filtered)

    # Load forecasts
    df_load = df.pivot(index='date', columns='hour', values='load_forecast')
    df_load = df_load.apply(lambda x: pd.to_numeric(x), axis=1)
    df_load = df_load.transpose().fillna(method='backfill').transpose()
    df_load = df_load.transpose().fillna(method='ffill').transpose()
    df_load = df_load.reindex(df_prices_log_filtered.index)

    # Temperatures
    df_temp = df.pivot(index='date', columns='hour', values='temp_dca')
    df_temp = df_temp.apply(lambda x: pd.to_numeric(x), axis=1)
    df_temp = df_temp.transpose().fillna(method='backfill').transpose()
    df_temp = df_temp.transpose().fillna(method='ffill').transpose()
    df_temp = df_temp.reindex(df_prices_log_filtered.index)

    holidays = USFederalHolidayCalendar().holidays(
        start='2011-01-01', end='2017-01-01').to_pydatetime()
    holiday_dates = set([h.date() for h in holidays])

    s = df_prices_log_filtered.reset_index()['date']
    data={"weekend": s.apply(lambda x: x.isoweekday() >= 6).values,
          "holiday": s.apply(lambda x: x in holiday_dates).values,
          "dst": s.apply(lambda x: tz.localize(
            dt.combine(x, dt.min.time())).dst().seconds > 0).values,
          "cos_doy": s.apply(lambda x: np.cos(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values,
          "sin_doy": s.apply(lambda x: np.sin(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values}
    df_feat = pd.DataFrame(data=data, index=df_prices_log_filtered.index)

    X = np.hstack([df_prices_log_filtered.iloc[:-1].values,        # past lmp
                    df_load.iloc[1:].values,        # future load forecast
                    df_temp.iloc[:-1].values,       # past temp
                    df_temp.iloc[:-1].values**2,    # past temp^2
                    df_temp.iloc[1:].values,        # future temp
                    df_temp.iloc[1:].values**2,     # future temp^2
                    df_temp.iloc[1:].values**3,     # future temp^3
                    df_feat.iloc[1:].values]).astype(np.float64)

    X[:,:] = \
        (X[:,:] - np.mean(X[:,:], axis=0)) / np.std(X[:,:], axis=0)

    Y = df_prices_log_filtered.iloc[1:].values

    return X, Y


def get_train_test_split(params, train_frac):
    X, Y = get_features_labels(params)

    n_tt = int(X.shape[0] * 0.8)
    X_train, Y_train = X[:n_tt, :], Y[:n_tt, :]
    X_test, Y_test   = X[n_tt:, :], Y[n_tt:, :]

    arrays = {'X_train': torch.Tensor(X_train), 'Y_train': torch.Tensor(Y_train), 
            'X_test': torch.Tensor(X_test), 'Y_test': torch.Tensor(Y_test)}

    return arrays

def get_loaders_tt(arrays_dict, bsz):
    train_loader = DataLoader(TensorDataset(
        arrays_dict['X_train'], arrays_dict['Y_train']), shuffle=False, batch_size=bsz)
    test_loader  = DataLoader(TensorDataset(
        arrays_dict['X_test'], arrays_dict['Y_test']), shuffle=False, batch_size=bsz)
    return {'train': train_loader, 'test': test_loader}

def get_loaders_tth(arrays_dict, bsz):
    train_loader = DataLoader(TensorDataset(
        arrays_dict['X_train'], arrays_dict['Y_train']), shuffle=False, batch_size=bsz)
    test_loader  = DataLoader(TensorDataset(
        arrays_dict['X_test'], arrays_dict['Y_test']), shuffle=False, batch_size=bsz)
    hold_loader  = DataLoader(TensorDataset(
        arrays_dict['X_hold'], arrays_dict['Y_hold']), shuffle=False, batch_size=bsz)
    return {'train': train_loader, 'test': test_loader, 'hold': hold_loader}

def get_train_hold_split(tensors_dict, th_frac, save_folder):
    X_train = tensors_dict['X_train']
    Y_train = tensors_dict['Y_train']

    inds = np.random.permutation(X_train.size(0))

    with open(os.path.join(save_folder, 'th_split_permutation'), 'wb') as f:
        np.save(f, inds)

    train_inds = torch.LongTensor(inds[ :int(X_train.size(0) * th_frac)])
    hold_inds = torch.LongTensor(inds[int(X_train.size(0) * th_frac):])

    X_train2, X_hold2 = X_train[train_inds, :], X_train[hold_inds, :]
    Y_train2, Y_hold2 = Y_train[train_inds, :], Y_train[hold_inds, :]

    tensors_task = {'X_train': X_train2, 'Y_train': Y_train2, 
            'X_hold': X_hold2, 'Y_hold': Y_hold2,
            'X_test': tensors_dict['X_test'].clone(),
            'Y_test': tensors_dict['Y_test'].clone()}
    return tensors_task

if __name__=='__main__':
    main()


