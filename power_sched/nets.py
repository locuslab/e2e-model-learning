#/usr/bin/env python3

import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.cuda

import model_classes

import ipdb


def task_loss(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
            0.5 * (Y_sched - Y_actual)**2).mean(0)


def rmse_loss(mu_pred, Y_actual):
    return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()


# TODO: minibatching
def run_rmse_net(model, variables, X_train, Y_train):
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(1000):
        opt.zero_grad()
        model.train()
        train_loss = nn.MSELoss()(
            model(variables['X_train_'])[0], variables['Y_train_'])
        train_loss.backward()
        opt.step()

        model.eval()
        test_loss = nn.MSELoss()(
            model(variables['X_test_'])[0], variables['Y_test_'])

        print(i, train_loss.data[0], test_loss.data[0])

    model.eval()
    model.set_sig(variables['X_train_'], variables['Y_train_'])

    return model


# TODO: minibatching
def run_task_net(model, variables, params, X_train, Y_train, args):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    solver = model_classes.SolveScheduling(params)

    # For early stopping
    prev_min = 0
    hold_costs = []
    model_states = []
    num_stop_rounds = 20

    for i in range(1000):
        opt.zero_grad()
        model.train()
        mu_pred_train, sig_pred_train = model(variables['X_train_'])
        Y_sched_train = solver(mu_pred_train.double(), sig_pred_train.double())
        train_loss = task_loss(
            Y_sched_train.float(),variables['Y_train_'], params)
        train_loss.sum().backward()

        model.eval()
        mu_pred_test, sig_pred_test = model(variables['X_test_'])
        Y_sched_test = solver(mu_pred_test.double(), sig_pred_test.double())
        test_loss = task_loss(
            Y_sched_test.float(), variables['Y_test_'], params)

        mu_pred_hold, sig_pred_hold = model(variables['X_hold_'])
        Y_sched_hold = solver(mu_pred_hold.double(), sig_pred_hold.double())
        hold_loss = task_loss(
            Y_sched_hold.float(), variables['Y_hold_'], params)

        opt.step()

        print(i, train_loss.sum().data[0], test_loss.sum().data[0], 
            hold_loss.sum().data[0])

        with open(os.path.join(args.save, 'task_losses.txt'), 'a') as f:
            f.write('{} {} {} {}\n'.format(i, train_loss.sum().data[0], 
                test_loss.sum().data[0], hold_loss.sum().data[0]))


        # Early stopping
        hold_costs.append(hold_loss.sum().data[0])
        model_states.append(model.state_dict().copy())
        if i > 0 and i % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            if prev_min == hold_costs[idx]:
                model.eval()
                best_model = model_classes.Net(
                    X_train[:,:-1], Y_train, [200, 200])
                best_model.load_state_dict(model_states[idx])
                best_model.cuda()
                return best_model
            else:
                prev_min = hold_costs[idx]
                hold_costs = [prev_min]
                model_states = [model_states[idx]]

    return model


# TODO: minibatching
def eval_net(which, model, variables, params, save_folder):
    solver = model_classes.SolveScheduling(params)

    model.eval()
    mu_pred_train, sig_pred_train = model(variables['X_train_'])
    mu_pred_test, sig_pred_test = model(variables['X_test_'])

    if (which == "task_net"):
        mu_pred_hold, sig_pred_hold = model(variables['X_hold_'])

    # Eval model on rmse
    train_rmse = rmse_loss(mu_pred_train, variables['Y_train_'])
    test_rmse = rmse_loss(mu_pred_test, variables['Y_test_'])

    if (which == "task_net"):
        hold_rmse = rmse_loss(mu_pred_hold, variables['Y_hold_'])

    with open(
        os.path.join(save_folder, '{}_train_rmse'.format(which)), 'wb') as f:
        np.save(f, train_rmse)

    with open(
        os.path.join(save_folder, '{}_test_rmse'.format(which)), 'wb') as f:
        np.save(f, test_rmse)

    if (which == "task_net"):
        with open(
            os.path.join(save_folder, '{}_hold_rmse'.format(which)), 'wb') as f:
            np.save(f, hold_rmse)

    # Eval model on task loss
    Y_sched_train = solver(mu_pred_train.double(), sig_pred_train.double())
    train_loss_task = task_loss(
        Y_sched_train.float(), variables['Y_train_'], params)

    Y_sched_test = solver(mu_pred_test.double(), sig_pred_test.double())
    test_loss_task = task_loss(
        Y_sched_test.float(), variables['Y_test_'], params)

    if (which == "task_net"):
        Y_sched_hold = solver(mu_pred_hold.double(), sig_pred_hold.double())
        hold_loss_task = task_loss(
            Y_sched_hold.float(), variables['Y_hold_'], params)

    # torch.save(train_loss_task.data[0], 
    #     os.path.join(save_folder, '{}_train_task'.format(which)))
    # torch.save(test_loss_task.data[0], 
    #     os.path.join(save_folder, '{}_test_task'.format(which)))
    torch.save(train_loss_task.data, 
        os.path.join(save_folder, '{}_train_task'.format(which)))
    torch.save(test_loss_task.data, 
        os.path.join(save_folder, '{}_test_task'.format(which)))

    if (which == "task_net"):
        # torch.save(hold_loss_task.data[0], 
        #     os.path.join(save_folder, '{}_hold_task'.format(which)))
        torch.save(hold_loss_task.data, 
            os.path.join(save_folder, '{}_hold_task'.format(which)))
