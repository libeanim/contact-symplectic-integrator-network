import sys
sys.path.append('../')

"""
Newton Cradle Experiment
------------------------

This is the setup used in the thesis
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from environments import NewtonCradle
from models import CDLNetwork, ResNet
from utils import TRAIN, PREDICT, RMSE

env = None
cdl_model = None
resnet = None


def run():
    global env, cdl_model, resnet, cdl_data, resnet_data
    env = NewtonCradle(steps=500, dt=0.01, epochs=3000)
    env.generate()

    # CD-LAGRANGE
    cdl_model = TRAIN(env, CDLNetwork, name='CDL')

    # RESNET
    resnet = TRAIN(env, ResNet, name='ResNet')

    cdl_data = PREDICT(env, cdl_model)
    resnet_data = PREDICT(env, resnet)


def plot_trajectory(savefig=False):
    # bottom_right=(0.3, -2.1)
    top_right=(0.25, 1.95)
    
    plt.figure(figsize=(14, 10))
    plt.subplot(2,2,1); plt.title('Ground Truth')
    plt.plot(env.trajectory[:, 0], env.trajectory[:, 2], 'o--', label='1')
    plt.plot(env.trajectory[:, 1], env.trajectory[:, 3], 'o--', label='2')
    plt.legend()

    plt.subplot(2,2,2); plt.title('CD-Lagrange')
    rmse_cdl = RMSE(env, cdl_data)
    plt.text(top_right[0], top_right[1], 'RMSE={:.3f}'.format(rmse_cdl))
    plt.plot(cdl_data[:, 0], cdl_data[:, 2], 'o--', label='1')
    plt.plot(cdl_data[:, 1], cdl_data[:, 3], 'o--', label='2')
    plt.legend()

    plt.subplot(2,2,3); plt.title('ResNet')
    rmse_resnet = RMSE(env, resnet_data)
    plt.text(top_right[0], top_right[1], 'RMSE={:.3f}'.format(rmse_resnet))
    plt.plot(resnet_data[:, 0], resnet_data[:, 2], 'o--', label='1')
    plt.plot(resnet_data[:, 1], resnet_data[:, 3], 'o--', label='2')
    plt.legend()
    if savefig:
        plt.savefig(env.get_filename('trajectory'))
    plt.show()


def plot_potential(savefig=False):
    t = np.arange(cdl_data.shape[0]) * env.dt
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.title('Gradient of Potential')
    pt = cdl_model.grad_potential(cdl_data[:, 0:2])
    plt.plot(t, pt[:, 0], 'b', label='CD-Lagrange, Right')
    plt.plot(t, pt[:, 1], 'r', label='CD-Lagrange, Left')

    # plt.plot(t, env.g * np.sin(cdl_data[:, 0]), 'b--', label='Ground truth, Right')
    # plt.plot(t, env.g * np.sin(cdl_data[:, 1]), 'r--', label='Ground truth, Left')
    plt.plot(t[:-1], env.g * np.sin(env.trajectory[:, 0]), 'b--', label='Ground truth, Right')
    plt.plot(t[:-1], env.g * np.sin(env.trajectory[:, 1]), 'r--', label='Ground truth, Left')
    plt.legend()
    plt.xlabel('Time in s')
    plt.subplot(1, 2, 2)
    plt.title('Contact function')
    ct = cdl_model.contact(tf.concat([cdl_data[1:, 0:2], cdl_data[:-1, 2:4]], 1))
    plt.plot(t[1:], ct, label='CD-Lagrange')
    dst, dt1, dt2 = (env.trajectory[:, 0] - env.trajectory[:, 1]) <= 0, env.trajectory[:, 2] <= 0, env.trajectory[:, 3] >= 0
    ctc = np.logical_and(dst, np.logical_and(dt1, dt2))
    plt.plot(t[1:], ctc, 'kx', label='Ground truth')
    plt.legend()
    plt.xlabel('Time in s')
    if savefig:
        plt.savefig(env.get_filename('potential'))
    plt.show()

def plot_loss(savefig=False):
    plt.figure(figsize=(10,8))
    plt.semilogy(cdl_model.loss_data[:, 1], label="CD-Lagrange")
    plt.semilogy(resnet.loss_data[:, 1], label="ResNet")
    plt.xlabel('Epochs'); plt.ylabel('MSE loss')
    plt.legend()
    if savefig:
        plt.savefig(env.get_filename('loss'))
    plt.show()