#!/usr/bin/env python3

import os

import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import seaborn as sns

def plot_results(save_folder, true_model):

    filename = os.path.join(save_folder, 
        'inventory_results_{}.csv'.format(true_model))

    with open(filename, 'r') as f:
        mle_true_score = float(f.readline().split(',')[1])
    df = pd.read_csv(filename, index_col=0, skiprows=1, na_values=['None'], 
        dtype=float)
    df['mle_true'] = mle_true_score
    df = df[['task-linear', 'task-nonlinear', 'policy', 'mle', 'mle_true']]

    # Means and std deviations of losses for each model and training set size
    means = df.groupby(df.index).mean()
    stds = df.groupby(df.index).std()

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(4, 3)

    styles = [ '-', '-.', '--', ':' , '-']
    colors = [sns.color_palette()[i] for i in [3,1,2,4,0]]

    ax.set_axis_bgcolor("none")

    for col, style, color in zip(means.columns, styles, colors):
        if col == 'mle_true':
            means[col].plot(ax=ax, lw=2, style=style, color=color)
        else:
            means[col].plot(ax=ax, lw=2, fmt=style, color=color, yerr=stds)

    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Inventory Stock Cost')

    y_min = (means.min().min() - 10).round(-1)
    y_max = (means.max().max() + 10).round(-1)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    ax.margins(0,0)
    ax.grid(linestyle=':', linewidth='0.5', color='gray')

    legend = ax.legend(loc='center left', bbox_to_anchor=(-0.2, -0.4), 
        shadow=False, ncol=5, fontsize=6.5, borderpad=0, frameon=False)

    fig.savefig("{}.pdf".format(filename[:-4]), dpi=100, encoding='pdf')
