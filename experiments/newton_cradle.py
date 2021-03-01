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
from models import CDLNetwork, ResNet, ResNetContact
from utils import TRAIN, PREDICT, RMSE

env = None
cdl_model = None
resnet = None
resnet_c = None


def run():
    global env, cdl_model, resnet, resnet_c, cdl_data, resnet_data, resnet_c_data
    env = NewtonCradle(steps=540, dt=0.01, epochs=2000, SIGMA=0.02)
    env.generate()
    env.plot()
    
    # CD-LAGRANGE 
    cdl_model = TRAIN(env, CDLNetwork, name='CDL') 

    # RESNET 
    resnet_c = TRAIN(env, ResNetContact, name='ResNetContact') 
    resnet = TRAIN(env, ResNet, name='ResNet') 

    cdl_data = PREDICT(env, cdl_model)
    resnet_data = PREDICT(env, resnet)
    resnet_c_data = PREDICT(env, resnet_c) 


plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans'})

def plot_trajectory(savefig=False):

    bottom_right = (0.3, -2.1)
    top_right = (0.25, 1.95)
    rmse_pos = bottom_right
    
    plt.figure(figsize=(8, 6.6))
    plt.subplot(2,2,1); plt.title('Ground Truth')
    plt.plot(env.trajectory[:, 0], env.trajectory[:, 2], '--', label='1')
    plt.plot(env.trajectory[:, 1], env.trajectory[:, 3], '--', label='2')

    plt.plot(env.X[:, 0], env.X[:, 2], 'C0x', label='Data 1', alpha=0.3)
    plt.plot(env.X[:, 1], env.X[:, 3], 'C1x', label='Data 2', alpha=0.3)
#     plt.legend()

    plt.subplot(2,2,2); plt.title('CD-Lagrange')
    rmse_cdl = RMSE(env, cdl_data)
    print('CDL RMSE={:.3f}'.format(rmse_cdl))
    plt.plot(cdl_data[:, 0], cdl_data[:, 2], 'x--', label='1')
    plt.plot(cdl_data[:, 1], cdl_data[:, 3], 'x--', label='2')
#     plt.legend()

    plt.subplot(2,2,3); plt.title('ResNet')
    rmse_resnet = RMSE(env, resnet_data)
    print('Resnet RMSE={:.3f}'.format(rmse_resnet))
    plt.plot(resnet_data[:, 0], resnet_data[:, 2], 'x--', label='1')
    plt.plot(resnet_data[:, 1], resnet_data[:, 3], 'x--', label='2')
#     plt.legend()

    plt.subplot(2,2,4); plt.title('ResNet Contact')
    rmse_resnet_c = RMSE(env, resnet_c_data)
    print('ResnetContact RMSE={:.3f}'.format(rmse_resnet_c))
    plt.plot(resnet_c_data[:, 0], resnet_c_data[:, 2], 'x--', label='1')
    plt.plot(resnet_c_data[:, 1], resnet_c_data[:, 3], 'x--', label='2')
#     plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(env.get_filename('trajectory'), bbox_inches="tight")
    plt.show()
    
def plot_potential(savefig=False):
    t = np.arange(cdl_data.shape[0]) * env.dt
    plt.figure(figsize=(4, 3.3*(2/3)))
    plt.subplot(1, 2, 1)
    plt.title('Gradient of Potential')
    pt = cdl_model.grad_potential(cdl_data[:, 0:2])
    plt.plot(t, pt[:, 0], 'b', label='CD-Lagrange, 1')
    plt.plot(t, pt[:, 1], 'r', label='CD-Lagrange, 2')
    plt.plot(t[:-1], env.g * np.sin(env.trajectory[:, 0]), 'b--', label='Ground truth, 1')
    plt.plot(t[:-1], env.g * np.sin(env.trajectory[:, 1]), 'r--', label='Ground truth, 2')
    plt.xlabel('Time in s')
    plt.subplot(1, 2, 2)
    plt.title('Contact function')
    
    dst, dt1, dt2 = (env.trajectory[:, 0] - env.trajectory[:, 1]) <= 0, env.trajectory[:, 2] <= 0, env.trajectory[:, 3] >= 0
    ctc = np.logical_and(dst, np.logical_and(dt1, dt2))
    plt.plot(t[1:], ctc, 'kx', label='Ground truth', alpha=0.4)
    
    plt.plot(t, resnet_c.contact(resnet_c_data[:, :-1]), 'C2', label='ResnetContact', linewidth=2.)
    ct = cdl_model.contact(tf.concat([cdl_data[1:, 0:2], cdl_data[:-1, 2:4]], 1))
    plt.plot(t[1:], ct, 'C0', label='CD-Lagrange', linewidth=2.)
    plt.xlabel('Time in s')
    plt.yticks([0, 1])
    if savefig:
        plt.savefig(env.get_filename('potential'), bbox_inches="tight")
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