import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy
import concurrent.futures
from Scripts.crocker import *

def compute_crocker_error(true_metric, pred_metric):

    if len(true_metric.shape) > 2:
    
        max_B0 = np.max(true_metric[:,:,0])
        true_B0 = true_metric[:,:,0]/max_B0
        pred_B0 = pred_metric[:,:,0]/max_B0

        max_B1 = np.max(true_metric[:,:,1])
        true_B1 = true_metric[:,:,1]/max_B1
        pred_B1 = pred_metric[:,:,1]/max_B1

        loss = np.sum(np.abs(true_B0-pred_B0)) + np.sum(np.abs(true_B1-pred_B1))
    else:
        loss = np.sum((np.log10(true_metric)-np.log10(pred_metric))**2/np.max(np.log10(true_metric))**2)

    return loss

betti_numbers = [0, 1]
#VANILLA CROCKER
#Which DataFrame columns to use as dimensions
DATA_COLS = ('x','y','angle')

#List of frame values to use, must be aligned for direct comparison
true_FRAME_LIST = range(20,120,1)
pred_FRAME_LIST = range(10,120,1) #starts at 10 because of angle computation


max_NUM_SAMPLE = int(1e4)
chosen_NUM_SAMPLE = int(1e4)

#Initial conditions
stage_idx = 37

#compute the data for the crocker plot
if 'angle' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
    true_crocker_path = 'Results/stage_'+str(stage_idx)+'/stage_'+str(stage_idx)+'_subsampled_crocker_angles.npy'
if 'vx' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
    true_crocker_path = 'Results/stage_'+str(stage_idx)+'/stage_'+str(stage_idx)+'_subsampled_crocker_velocities.npy'
true_crocker = np.load(true_crocker_path)

sampled_idc = []
sampled_pars = []
sampled_losses = []

for sampled_idx in range(chosen_NUM_SAMPLE):
    sample_dir = './Simulated_Grid/ODE/stage_'+str(stage_idx)+'/sample_'+str(max_NUM_SAMPLE)+'/run_'+str(sampled_idx+1)+'/'
    
    if os.path.isdir(sample_dir):
        #compute the data for the crocker plot
        if 'angle' in DATA_COLS:
            sampled_crocker_path = sample_dir+'crocker_angles.npy'
            save_loss_path = sample_dir+'loss_angles.npy'
        if 'vx' in DATA_COLS:
            sampled_crocker_path = sample_dir+'crocker_velocities.npy'
            save_loss_path = sample_dir+'loss_velocities.npy'
        
        if os.path.isfile(sampled_crocker_path):
            sampled_crocker = np.load(sampled_crocker_path)
            loss = compute_crocker_error(true_crocker, sampled_crocker)
            np.save(save_loss_path,loss)
            
#            if os.path.isfile(save_loss_path):
#                print(sampled_idx)
#                os.remove(sampled_crocker_path)
#
#                if os.path.isfile(sample_dir+'df.pkl'):
#                    os.remove(sample_dir+'df.pkl')

