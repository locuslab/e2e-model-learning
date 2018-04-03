import os
import pandas as pd
import numpy as np

def load_results(load_folders):
    rmse_loss_files = ['rmse_net_train_rmse', 'rmse_net_hold_rmse', 'rmse_net_test_rmse',
        'task_net_train_rmse', 'task_net_hold_rmse', 'task_net_test_rmse']
    task_loss_files = ['rmse_net_train_task', 'rmse_net_hold_task', 'rmse_net_test_task', 
        'task_net_train_task', 'task_net_hold_task', 'task_net_test_task']
    col_names = ['RMSE Net (train)', 'RMSE Net (hold)', 'RMSE Net (test)',
        'Task Net (train)', 'Task Net (hold)', 'Task Net (test)']

    df_rmse = pd.DataFrame()
    df_task = pd.DataFrame()
    for folder in load_folders:
        rmse_results, task_results = [], []
            
        for filename in rmse_loss_files:
            with open(os.path.join(folder, filename), 'rb') as f:
                rmse_results.append(np.load(f))
                
        df = pd.DataFrame(pd.DataFrame(rmse_results).T)
        df.columns = col_names
        df_rmse = df_rmse.append(df)

        for filename in task_loss_files:
            with open(os.path.join(folder, filename), 'rb') as f:
                task_results.append(np.load(f))
        
        df = pd.DataFrame(pd.DataFrame(task_results).T)
        df.columns = col_names
        df_task = df_task.append(df)

    return df_rmse, df_task

def get_means_stds(df):
    df2 = df.reset_index(drop=True)
    df2 = df2.groupby(df2.index // 24).sum() # aggregate daily data
    return df2.mean(), df2.std()

def calc_stats(load_folders, save_folder):
    df_rmse, df_task = load_results(load_folders)
    rmse_mean, rmse_stds = get_means_stds(df_rmse)
    task_mean, task_stds = get_means_stds(df_task)

    agg_stats = pd.concat([rmse_mean, rmse_stds, task_mean, task_stds], axis=1)
    agg_stats.columns = ['rmse (mean)', 'rmse (std)', 'task_loss (mean)', 'task_loss (std)']

    agg_stats.to_csv(os.path.join(save_folder, 'results.csv'))