#/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
import torch.cuda

import model_classes

import os

import numpy as np

import pickle

def task_loss(sched, Y_actual, params):
    T = params['T']
    z_in = sched[:, :T].float()
    z_out = sched[:, T:2*T].float()
    x = sched[:, 2*T:].float()
    costs = (
        (z_in - z_out) * Y_actual) + \
         (params['lambda'] * (x - params['B']/2)**2) + \
         (params['epsilon'] * z_in**2) + \
        (params['epsilon'] * z_out**2)
    return costs.mean(dim=0)


def rmse_loss(mu_pred, Y_actual):
    return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data


def run_rmse_net(model, loaders, params, tensors_task):
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # For early stopping
    prev_min = 0
    hold_costs = []
    model_states = []
    num_stop_rounds = 20

    for i in range(1000):

        # train
        model.train()
        total_train_loss = 0
        m_train = 0

        for (batch, (X_train, Y_train)) in enumerate(loaders['train']):
            X_train_, Y_train_ = (Variable(X_train.cuda()), Variable(Y_train.cuda()))

            opt.zero_grad()
            train_loss = nn.MSELoss()(model(X_train_), Y_train_)
            total_train_loss += train_loss.data[0] * X_train_.size(0)
            m_train += X_train_.size(0)
            train_loss.backward()
            opt.step()


        # evaluate on test
        model.eval()
        total_test_loss = 0
        m_test = 0

        for (batch, (X_test, Y_test)) in enumerate(loaders['test']):
            X_test_, Y_test_ = (Variable(X_test.cuda()), Variable(Y_test.cuda()))

            test_loss = nn.MSELoss()(model(X_test_), Y_test_)
            total_test_loss += test_loss.data[0] * X_test_.size(0)
            m_test += X_test_.size(0)

        model.eval()
        total_hold_loss = 0
        m_hold = 0
        for (batch, (X_hold, Y_hold)) in enumerate(loaders['hold']):
            X_hold_, Y_hold_ = (Variable(X_hold).cuda(), Variable(Y_hold).cuda())
            hold_loss = nn.MSELoss()(model(X_hold_), Y_hold_)
            total_hold_loss += hold_loss.data[0] * X_hold_.size(0)
            m_hold += X_hold_.size(0)

        print(i, total_train_loss/m_train, total_test_loss/m_test, total_hold_loss/m_hold)


        # Early stopping
        hold_costs.append(total_hold_loss)
        model_states.append(model.state_dict().copy())
        if i > 0 and i % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            if prev_min == hold_costs[idx]:
                model.eval()

                best_model = model_classes.Net(
                    tensors_task['X_train'], tensors_task['Y_train'], [200, 200], params['T'])
                best_model.load_state_dict(model_states[idx])
                best_model = best_model.cuda()

                return best_model
            else:
                prev_min = hold_costs[idx]
                hold_costs = [prev_min]
                model_states = [model_states[idx]]

    return model


def run_task_net(model, loader, params, args, tensors_task):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    solver = model_classes.ScheduleBattery(params)

    # For early stopping
    prev_min = 0
    hold_costs = []
    model_states = []
    num_stop_rounds = 20

    for i in range(1000):

        # train
        model.train()
        total_train_loss = 0
        m_train = 0
        for (batch, (X_train, Y_train)) in enumerate(loader['train']):
            opt.zero_grad()
            X_train, Y_train = (Variable(X_train).cuda(), Variable(Y_train).cuda())
            preds_train = model(X_train)
            train_loss = task_loss(solver(preds_train), Y_train, params).sum()
            total_train_loss += train_loss.data[0] * X_train.size(0)
            m_train += X_train.size(0)
            train_loss.backward()

        # test
        model.eval()
        total_test_loss = 0
        m_test = 0
        for (batch, (X_test, Y_test)) in enumerate(loader['test']):
            X_test, Y_test = (Variable(X_test).cuda(), Variable(Y_test).cuda())
            preds_test = model(X_test)
            test_loss = task_loss(solver(preds_test), Y_test, params).sum()
            total_test_loss += test_loss.data[0] * X_test.size(0)
            m_test += X_test.size(0)

        # hold
        model.eval()
        total_hold_loss = 0
        m_hold = 0
        for (batch, (X_hold, Y_hold)) in enumerate(loader['hold']):
            X_hold, Y_hold = (Variable(X_hold).cuda(), Variable(Y_hold).cuda())
            preds_hold = model(X_hold)
            hold_loss = task_loss(solver(preds_hold), Y_hold, params).sum()
            total_hold_loss += hold_loss.data[0] * X_hold.size(0)
            m_hold += X_hold.size(0)

        print(i, total_train_loss/m_train, total_test_loss/m_test, total_hold_loss/m_hold)

        # Early stopping
        hold_costs.append(total_hold_loss)
        model_states.append(model.state_dict().copy())
        if i > 0 and i % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            if prev_min == hold_costs[idx]:
                model.eval()

                best_model = model_classes.Net(
                    tensors_task['X_train'], tensors_task['Y_train'], [200, 200], params['T'])
                best_model.load_state_dict(model_states[idx])
                best_model = best_model.cuda()

                return best_model
            else:
                prev_min = hold_costs[idx]
                hold_costs = [prev_min]
                model_states = [model_states[idx]]

    return model


def eval_for_loaders(which, model, loaders, params, save_folder, loader_label):
    total_loss_rmse = torch.zeros(params['T']).cuda()
    total_loss_task = torch.zeros(params['T']).cuda()
    total_loss_task_2 = torch.zeros(params['T']).cuda()
    all_preds = torch.zeros(1, params['T']).cuda()

    solver = model_classes.ScheduleBattery(params).cuda()

    m = 0
    for (batch, (X, y)) in enumerate(loaders[loader_label]):
        X, y = (Variable(X.cuda()), Variable(y.cuda()))
        preds = model(X)

        all_preds = torch.cat([all_preds, preds.data], 0)

        total_loss_rmse += rmse_loss(preds, y) * X.size(0)

        sched = solver(preds)

        total_loss_task += task_loss(sched, y, params).data * X.size(0)

        m += X.size(0)

    with open(os.path.join(save_folder, '{}_{}_rmse'.format(which, loader_label)), 'wb') as f:
        np.save(f, total_loss_rmse.cpu().numpy()/m)

    with open(os.path.join(save_folder, '{}_{}_task'.format(which, loader_label)), 'wb') as f:
        np.save(f, total_loss_task.cpu().numpy()/m)

    with open(os.path.join(save_folder, '{}_{}_preds'.format(which, loader_label)), 'wb') as f:
        np.save(f, all_preds.cpu().numpy())

    torch.save(model.state_dict(), os.path.join(save_folder, '{}_model'.format(which)))


def eval_net(which, model, loaders, params, save_folder):
    model.eval()
    eval_for_loaders(which, model, loaders, params, save_folder, 'train')
    eval_for_loaders(which, model, loaders, params, save_folder, 'test')
    eval_for_loaders(which, model, loaders, params, save_folder, 'hold')

