#/usr/bin/env python3

import os
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib.ticker as ticker
import seaborn as sns

import torch

def load_results(load_folders):
    array_files = ['task_net_test_rmse', 'rmse_net_test_rmse', 'weighted_rmse_net_test_rmse']
    float_tensor_files = ['task_net_test_task', 'rmse_net_test_task', 'weighted_rmse_net_test_task']
    col_names = ['Task-based', 'RMSE', 'Cost-weighted RMSE']

    df_rmse = pd.DataFrame()
    df_task = pd.DataFrame()
    for folder in load_folders:
        arrays, tensors = [], []
            
        for filename in array_files:
            with open(os.path.join(folder, filename), 'rb') as f:
                arrays.append(np.load(f))
                
        df = pd.DataFrame(pd.DataFrame(arrays).T)
        df.columns = col_names
        df_rmse = df_rmse.append(df)

        for filename in float_tensor_files:
            tensors.append(torch.load(os.path.join(folder, filename)))
        
        df = pd.DataFrame(pd.DataFrame(tensors).T)
        df.columns = col_names
        df_task = df_task.append(df)

    return df_rmse, df_task

def get_means_stds(df):
    return df.groupby(df.index).mean(), df.groupby(df.index).std()

def plot_results(load_folders, save_folder):
    df_rmse, df_task = load_results(load_folders)
    rmse_mean, rmse_stds = get_means_stds(df_rmse)
    task_mean, task_stds = get_means_stds(df_task)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(8.5, 3)

    styles = [ '-', '--', '-.']
    colors = [sns.color_palette()[i] for i in [1,2]] + ['gray']

    ax = axes[0]
    ax.set_axis_bgcolor('none')
    for col, style, color in zip(rmse_mean.columns, styles, colors):
        rmse_mean[col].plot(
            ax=ax, lw=2, fmt=style, color=color, yerr=rmse_stds[col])
    ax.set_ylabel('RMSE')

    ax2 = axes[1]
    ax2.set_axis_bgcolor('none')
    for col, style, color in zip(task_mean.columns, styles, colors):
        if col == 'Cost-weighted RMSE':
            task_mean[col].plot(
                ax=ax2, lw=2, style=style, color=color)
            ax2.errorbar(task_mean.index+0.2, task_mean[col], 
                yerr=task_stds[col], color=color, lw=0, elinewidth=2)
        else:
            task_mean[col].plot(
                ax=ax2, lw=2, fmt=style, color=color, yerr=task_stds[col])
    ax2.set_ylabel('Task Loss')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    for a in [ax, ax2]:
        a.margins(0,0)
        a.grid(linestyle=':', linewidth='0.5', color='gray')
        a.xaxis.set_major_locator(ticker.MultipleLocator(4))
        a.set_xlim(0, 24)
        a.set_ylim(0, )

    # Joint x-axis label and legend
    fig.text(0.5, 0.13, 'Hour of Day', ha='center', fontsize=12)
    legend = ax.legend(loc='center left', bbox_to_anchor=(0.3, -0.4), 
        shadow=False, ncol=7, fontsize=12, borderpad=0, frameon=False)

    fig.savefig(os.path.join(save_folder, 
        '{}.pdf'.format(save_folder)), dpi=100, encoding='pdf')
