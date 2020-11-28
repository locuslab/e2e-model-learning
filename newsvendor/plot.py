#!/usr/bin/env python3

import os

import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import seaborn as sns

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

def plot_results(save_folder, true_model):

    filename = os.path.join(save_folder, 'inventory_results.csv')

    with open(filename, 'r') as f:
        mle_true_score = float(f.readline().split(',')[1])
    df = pd.read_csv(filename, index_col=0, skiprows=1, na_values=['None'], 
        dtype=float)
    df['mle_true'] = mle_true_score

    df_linear    =  df[['mle_true', 'task-linear', 'mle-linear', 'policy-linear']]
    df_nonlinear =  df[['mle_true', 'task-nonlinear', 'mle-nonlinear', 'policy-nonlinear']]

    # Means and std deviations of losses for each model and training set size
    means_linear = df_linear.groupby(df_linear.index).mean()
    stds_linear  = df_linear.groupby(df_linear.index).std()

    means_nonlinear = df_nonlinear.groupby(df_nonlinear.index).mean()
    stds_nonlinear  = df_nonlinear.groupby(df_nonlinear.index).std()

    fig, axes = plt.subplots(1,2, sharey=True)
    fig.set_size_inches(8.5, 2.5)

    styles = ['--', '-', ':', '-.']
    colors = ['gray'] + [sns.color_palette(n_colors=8)[i] for i in [1, 4, 2]]

    # For size of error bars
    capsize = 2
    capthick = 1

    ax = axes[0]
    for col, style, color in zip(means_linear.columns, styles, colors):
        if col == 'mle_true':
            means_linear[col].plot(ax=ax, lw=2, style=style, color=color)
        else:
            means_linear[col].plot(ax=ax, lw=2, fmt=style, color=color, yerr=stds_linear, 
                capsize=capsize, capthick=capthick)
    ax.set_xlabel('linear hypothesis')
    ax.set_ylabel('Inventory Stock Cost')
    ax.set_ylim(mle_true_score-5, )
    ax.grid(linestyle=':', linewidth='0.5', color='gray')

    ax = axes[1]
    for col, style, color in zip(means_nonlinear.columns, styles, colors):
        if col == 'mle_true':
            means_nonlinear[col].plot(ax=ax, lw=2, style=style, color=color)
        else:
            means_nonlinear[col].plot(ax=ax, lw=2, fmt=style, color=color, yerr=stds_nonlinear, 
                capsize=capsize, capthick=capthick)
    ax.set_xlabel('nonlinear hypothesis')
    ax.set_ylabel('Inventory Stock Cost')
    ax.grid(linestyle=':', linewidth='0.5', color='gray')


    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Inventory Stock Cost')

    legend = ax.legend(labels = ['True Params', 'Task-based (our method)', 'MLE', 'Policy Optimizer'],
        loc='center left', bbox_to_anchor=(-0.2, -0.4), 
        shadow=False, ncol=5, fontsize=6.5, borderpad=0, frameon=False)

    fig.savefig("{}.pdf".format(filename[:-4]), dpi=100, encoding='pdf')
