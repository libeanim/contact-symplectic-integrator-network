"""
Evaluation utils file
=====================

The functions in this file help to evaluate the performance of a model
and its ability to generalise.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import PREDICT

def RMSE(env, pred):
    """
    Calculate the root-mean-squared-error for a given environment and prediction
    
    Parameters
    ----------
        env:  Envrionment
        pred: Predicted data
    """
    return np.sqrt(np.sum(pred[:-1] - env.trajectory)**2 / env.trajectory.shape[0])

def calculate_RMSE(exp, state0s, enable_vin=False):
    """
    Calculate the root-mean-squared-error for a given experiment and a list of start states

    Parameters
    ----------
        exp: Experiment
        state0s: list of start states
        enable_vin: bool; also calculate the results of the VIN
                    (this requires a trained VIN in the experiment)
    """
    rmses = []
    for y in state0s:
        exp.env.generate(state0=y)
        cdle = PREDICT(exp.env, exp.cdl_model).numpy()
        rese = PREDICT(exp.env, exp.resnet).numpy()
        if enable_vin:
            vine = PREDICT(exp.env, exp.vin_model).numpy()
            rmses.append((RMSE(exp.env, cdle), RMSE(exp.env, rese), RMSE(exp.env, vine)))
        else:
            rmses.append((RMSE(exp.env, cdle), RMSE(exp.env, rese)))
    return np.array(rmses)

def get_random_start_states(multiplier=1, dim_state=2, length=10, SEED=0):
    """Generate a list of random start states"""
    np.random.seed(SEED)
    return np.random.rand(length, dim_state) * multiplier

def plot_RMSE(rmses):
    """Plot the root-mean-squared-error over multiple start states"""
    plt.semilogy(rmses[:, 0], label='CDL')
    plt.semilogy(rmses[:, 1], label='ResNet')
    if rmses.shape[1] == 2:
        plt.semilogy(rmses[:, 2], label='VIN')
    plt.legend()
    plt.show()
    return rmses