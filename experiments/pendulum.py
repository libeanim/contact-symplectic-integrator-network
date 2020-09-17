"""
Pendulum Experiment
--------------------

This is the setup used in the thesis
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from environments import Pendulum
from models import CDLNetwork_Simple, ResNet, VIN_VV
from utils import TRAIN, PREDICT, RMSE

env = None
cdl_model = None
resnet = None

_train_vin = False

def run(train_vin=False):
    global env, cdl_model, resnet, vin_model, cdl_data, resnet_data, vin_data, _train_vin
    _train_vin = train_vin
    env = Pendulum(steps=400, dt=0.01, epochs=3000, friction=0., length=1, SIGMA=0.1)
    env.generate()

    # CD-LAGRANGE
    cdl_model = TRAIN(env, CDLNetwork_Simple, name='CDL')

    # RESNET
    resnet = TRAIN(env, ResNet, name='ResNet')

    cdl_data = PREDICT(env, cdl_model)
    resnet_data = PREDICT(env, resnet)

    if _train_vin:
        # VIN
        vin_model = TRAIN(env, VIN_VV, name='VIN_VV')
        vin_data = PREDICT(env, vin_model)


def plot_trajectory(savefig=False):
    ## Phase space
    plt.figure(figsize=(12,8))
    plt.title('Phase Space'); plt.xlabel('q'); plt.ylabel('p')
    plt.plot(env.trajectory[:, 0], env.trajectory[:, 1], '--k', label='Ground truth')
    plt.plot(env.X[:, 0], env.X[:, 1], 'xk', label='Training data', alpha=0.3)
    plt.plot(cdl_data[:, 0], cdl_data[:, 1], '-',  label='CD-Lagrange, RMSE: {:.3f}'.format(RMSE(env, cdl_data)))
    plt.plot(resnet_data[:, 0], resnet_data[:, 1], '-', label='ResNet, RMSE: {:.3f}'.format(RMSE(env, resnet_data)))
    if _train_vin:
        plt.plot(vin_data[:, 0], vin_data[:, 1], '-', label='VIN VV, RMSE: {:.3f}'.format(RMSE(env, vin_data)))
    plt.legend()
    if savefig:
        plt.savefig(env.get_filename('trajectory'))
    plt.show()

def plot_energy(savefig=False):
    ## Energy
    e_exact = env.energy()
    e_cdl = env.energy(cdl_data)
    e_resnet = env.energy(resnet_data)
    if _train_vin:
        e_vv = env.energy(vin_data)
        e_vv = e_vv[:-1]
    e_cdl, e_resnet = e_cdl[:-1], e_resnet[:-1]

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title('Energy'); plt.xlabel('t')
    plt.plot(e_exact, '--k', label='Ground truth')
    plt.plot(e_cdl, '-', label='CD-Lagrange')
    plt.plot(e_resnet, '-', label='ResNet')
    if _train_vin:
        plt.plot(e_vv, '-', label='VIN VV')
    plt.legend()
    plt.subplot(1,2,2)
    plt.title('Energy Error'); plt.xlabel('t')
    plt.plot(e_exact - e_exact, '--k', label='Ground truth')
    plt.plot(e_cdl - e_exact, label='CD-Lagrange')
    plt.plot(e_resnet - e_exact, label='ResNet')
    if _train_vin:
        plt.plot(e_vv - e_exact, label='VIN VV')
    if savefig:
        plt.savefig(env.get_filename('energy'))
    plt.show()


def plot_potential(savefig=False):
    t = np.arange(cdl_data.shape[0]) * env.dt

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.title('Gradient of Potential')
    plt.plot(t, cdl_model.grad_potential(tf.reshape(cdl_data[:, 0], (cdl_data.shape[0], 1)))[:, 0], label='CD-Lagrange')
    # plt.plot(t, env.g * np.sin(cdl_data[:, 0]), '--k', label='Ground truth')
    plt.plot(t[:-1], env.g * np.sin(env.trajectory[:, 0]), '--k', label='Ground truth')
    if _train_vin:
        plt.plot(t, -vin_model.grad_potential(tf.reshape(vin_data[:, 0], (vin_data.shape[0], 1)))[:, 0], 'C2', label='VIN VV')
    plt.xlabel('Time in s')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Contact function')
    plt.plot(t, np.zeros_like(t), '--k', label='Ground truth')
    plt.plot(t[1:], cdl_model.contact(tf.concat([cdl_data[1:, 0:1], cdl_data[:-1, 1:2]], 1))[:, 0], label='CD-Lagrange')
    plt.ylim((-0.1, 1))
    plt.xlabel('Time in s')
    plt.legend()
    if savefig:
        plt.savefig(env.get_filename('potential'))
    plt.show()

def plot_loss(savefig=False):
    plt.figure(figsize=(10,8))
    plt.semilogy(cdl_model.loss_data[:, 1], label="CD-Lagrange")
    plt.semilogy(resnet.loss_data[:, 1], label="ResNet")
    if _train_vin:
        plt.semilogy(vin_model.loss_data[:, 1], label="VIN VV")
    plt.xlabel('Epochs'); plt.ylabel('MSE loss')
    plt.legend()
    if savefig:
        plt.savefig(env.get_filename('loss'))
    plt.show()