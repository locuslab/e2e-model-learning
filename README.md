# Task-based End-to-end Model Learning in Stochastic Optimization

This repository is by 
[Priya L. Donti](https://www.priyadonti.com),
[Brandon Amos](http://bamos.github.io),
and [J. Zico Kolter](http://zicokolter.com)
and contains the [PyTorch](https://pytorch.org) source code to
reproduce the experiments in our paper
[Task-based End-to-end Model Learning in Stochastic Optimization](https://arxiv.org/abs/1703.04529).

If you find this repository helpful in your publications,
please consider citing our paper.

```
@inproceedings{donti2017task,
  title={Task-based end-to-end model learning in stochastic optimization},
  author={Donti, Priya and Amos, Brandon and Kolter, J Zico},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5484--5494},
  year={2017}
}
```

# Introduction

As machine learning techniques have become more ubiquitous, it has become 
common to see machine learning prediction algorithms operating within some 
larger process. However, the criteria by which we train machine learning 
algorithms often differ from the ultimate criteria on which we evaluate them.

This repository demonstrates an end-to-end approach for learning probabilistic 
machine learning models within the context of stochastic programming, in a 
manner that directly captures the ultimate task-based objective for which they 
will be used. Specifically, we evaluate our approach in the context of
(a) a generic inventory stock problem and (b) an electrical grid scheduling
task based on over eight years of data from PJM.

Please see our paper [Task-based End-to-end Model Learning in Stochastic Optimization](https://arxiv.org/abs/1703.04529)
and the code in this repository ([locuslab/e2e-model-learning](https://github.com/locuslab/e2e-model-learning))
for more details about the general approach proposed and our initial
experimental implementations.


## Setup and Dependencies

+ Python 3.x/numpy/scipy
+ [cvxpy](http://www.cvxpy.org/en/latest/) 1.x
+ [PyTorch](https://pytorch.org) 1.x
+ [qpth](https://github.com/locuslab/qpth) 0.0.15:
  *A fast QP solver for PyTorch released in conjunction with the paper 
  [OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443).*
+ [bamos/block](https://github.com/bamos/block):
  *An intelligent block matrix library for numpy, PyTorch, and beyond.*
+ pandas/matplotlib/seaborn
+ Optional: [bamos/setGPU](https://github.com/bamos/setGPU):
  A small library to set `CUDA_VISIBLE_DEVICES` on multi-GPU systems.
+ Optional: setproctitle: A library to set process names.

# Inventory Stock Problem (Newsvendor) Experiments

Experiments considering a "conditional" variation of the inventory stock problem. 
Problem instances are generated via random sampling.

```
newsvendor
├── main.py - Run inventory stock problem experiments. (See arguments.)
├── task_net.py - Functions for our task-based end-to-end model learning approach.
├── mle.py - Functions for linear maximum likelihood estimation approach.
├── mle_net.py - Functions for nonlinear maximum likelihood estimation approach.
├── policy_net.py - Functions for end-to-end neural network policy model.
├── batch.py - Helper functions for minibatched evaluation.
├── plot.py - Plot experimental results.
└── constants.py - Constants to set GPU vs. CPU.
```

# Load Forecasting and Generator Scheduling Experiments

Experiments considering a realistic grid-scheduling task, in which
electricity generation is scheduled based on some (unknown) distribution
over electricity demand. Historical load data for these experiments were obtained from
[PJM](http://www.pjm.com/markets-and-operations/ops-analysis/historical-load-data.aspx).

```
power_sched
├── main.py - Run load forecasting problem experiments. (See arguments.)
├── model_classes.py - Models used for experiments.
├── nets.py - Functions for RMSE, cost-weighted RMSE, and task nets.
├── plot.py - Plot experimental results.
├── constants.py - Constants to set GPU vs. CPU.
└── pjm_load_data_*.txt - Historical load data from PJM.
```

# Price Forecasting and Battery Storage Experiments

Experiments considering a realistic battery arbitrage task, in which
a power grid-connected battery generates a charge/discharge schedule 
based on some (unknown) distribution
over energy prices. Historical energy price data for these experiments were obtained from
[PJM](http://www.pjm.com/markets-and-operations/energy/real-time/monthlylmp.aspx).

```
battery_storage
├── main.py - Run battery storage problem experiments. (See arguments.)
├── model_classes.py - Models used for experiments.
├── nets.py - Functions for RMSE and task nets.
├── calc_stats.py - Calculate experimental result stats.
├── constants.py - Constants to set GPU vs. CPU.
└── storage_data.csv - Historical energy price data from PJM.
```

### Acknowledgments

This material is based upon work supported by the 
National Science Foundation Graduate Research Fellowship Program under
Grant No. DGE1252522. 

# Licensing

Unless otherwise stated, the source code is copyright
Carnegie Mellon University and licensed under the
[Apache 2.0 License](./LICENSE).
