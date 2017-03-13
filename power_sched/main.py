#/usr/bin/env python3

import argparse

import os
import pandas as pd
import numpy as np

from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

try: import setGPU
except ImportError: pass

import torch
from torch.autograd import Variable, Function
import torch.cuda

import setproctitle

import model_classes, nets, plot

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

def main():
    parser = argparse.ArgumentParser(
        description='Run electricity scheduling task net experiments.')
    parser.add_argument('--save', type=str, required=True, 
        metavar='save-folder', help='save folder path')
    parser.add_argument('--nRuns', type=int, default=10,
        metavar='runs', help='number of runs')
    args = parser.parse_args()

    setproctitle.setproctitle('pdonti.' + args.save)

    X1, Y1 = load_data_with_features('pjm_load_data_2008-11.txt')
    X2, Y2 = load_data_with_features('pjm_load_data_2012-16.txt')

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    # Train, test split.
    n_tt = int(len(X) * 0.8)
    X_train, Y_train = X[:n_tt], Y[:n_tt]
    X_test, Y_test = X[n_tt:], Y[n_tt:]

    # Construct tensors (without intercepts).
    X_train_ = Variable(torch.Tensor(X_train[:,:-1])).cuda()
    Y_train_ = Variable(torch.Tensor(Y_train)).cuda()
    X_test_ = Variable(torch.Tensor(X_test[:,:-1])).cuda()
    Y_test_ = Variable(torch.Tensor(Y_test)).cuda()
    variables_rmse = {'X_train_': X_train_, 'Y_train_': Y_train_, 
                'X_test_': X_test_, 'Y_test_': Y_test_}

    for run in range(args.nRuns):

        save_folder = os.path.join(args.save, str(run))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Generation scheduling problem params.
        params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}

        # Run and eval rmse-minimizing net
        model_rmse = model_classes.Net(X_train[:,:-1], Y_train, [200, 200])
        model_rmse.cuda()
        model_rmse = nets.run_rmse_net(
            model_rmse, variables_rmse, X_train, Y_train)
        nets.eval_net("rmse_net", model_rmse, variables_rmse, params, save_folder)

        # Randomly construct hold-out set for task net training.
        th_frac = 0.8
        inds = np.random.permutation(X_train.shape[0])
        train_inds = inds[ :X_train.shape[0] * th_frac]
        hold_inds = inds[X_train.shape[0] * th_frac:]
        X_train2, X_hold2 = X_train[train_inds, :], X_train[hold_inds, :]
        Y_train2, Y_hold2 = Y_train[train_inds, :], Y_train[hold_inds, :]
        X_train2_ = Variable(torch.Tensor(X_train2[:,:-1])).cuda()
        Y_train2_ = Variable(torch.Tensor(Y_train2)).cuda()
        X_hold2_ = Variable(torch.Tensor(X_hold2[:,:-1])).cuda()
        Y_hold2_ = Variable(torch.Tensor(Y_hold2)).cuda()
        variables_task = {'X_train_': X_train2_, 'Y_train_': Y_train2_, 
                'X_hold_': X_hold2_, 'Y_hold_': Y_hold2_,
                'X_test_': X_test_, 'Y_test_': Y_test_}

        # Run and eval task-minimizing net, building off rmse net results.
        model_task = model_classes.Net(X_train2[:,:-1], Y_train2, [200, 200])
        model_task.cuda()
        model_task = nets.run_rmse_net(
            model_task, variables_task, X_train2, Y_train2)
        model_task = nets.run_task_net(
            model_task, variables_task, params, X_train2, Y_train2, args)
        nets.eval_net("task_net", model_task, variables_task, params, save_folder)

    plot.plot_results(map(
        lambda x: os.path.join(args.save, str(x)), range(args.nRuns)), args.save)


def load_data_with_features(filename):
    tz = pytz.timezone("America/New_York")
    df = pd.read_csv(filename, sep=" ", header=None, usecols=[1,2,3], 
        names=["time","load","temp"])
    df["time"] = df["time"].apply(dt.fromtimestamp, tz=tz)
    df["date"] = df["time"].apply(lambda x: x.date())
    df["hour"] = df["time"].apply(lambda x: x.hour)
    df.drop_duplicates("time", inplace=True)

    # Create one-day tables and interpolate missing entries
    df_load = df.pivot(index="date", columns="hour", values="load")
    df_temp = df.pivot(index="date", columns="hour", values="temp")
    df_load = df_load.transpose().fillna(method="backfill").transpose()
    df_load = df_load.transpose().fillna(method="ffill").transpose()
    df_temp = df_temp.transpose().fillna(method="backfill").transpose()
    df_temp = df_temp.transpose().fillna(method="ffill").transpose()

    holidays = USFederalHolidayCalendar().holidays(
        start='2008-01-01', end='2014-12-31').to_pydatetime()
    holiday_dates = set([h.date() for h in holidays])

    s = df_load.reset_index()["date"]
    data={"weekend": s.apply(lambda x: x.isoweekday() >= 6).values,
          "holiday": s.apply(lambda x: x in holiday_dates).values,
          "dst": s.apply(lambda x: tz.localize(
            dt.combine(x, dt.min.time())).dst().seconds > 0).values,
          "cos_doy": s.apply(lambda x: np.cos(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values,
          "sin_doy": s.apply(lambda x: np.sin(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values}
    df_feat = pd.DataFrame(data=data, index=df_load.index)

    # Construct features and normalize (all but intercept)
    X = np.hstack([df_load.iloc[:-1].values,        # past load
                    df_temp.iloc[:-1].values,       # past temp
                    df_temp.iloc[:-1].values**2,    # past temp^2
                    df_temp.iloc[1:].values,        # future temp
                    df_temp.iloc[1:].values**2,     # future temp^2
                    df_temp.iloc[1:].values**3,     # future temp^3
                    df_feat.iloc[1:].values,        
                    np.ones((len(df_feat)-1, 1))]).astype(np.float64)
    X[:,:-1] = \
        (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)

    Y = df_load.iloc[1:].values

    return X, Y


if __name__=='__main__':
    main()