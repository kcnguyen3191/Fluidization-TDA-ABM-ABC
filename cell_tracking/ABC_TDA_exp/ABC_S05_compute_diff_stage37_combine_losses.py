import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy
import concurrent.futures
from Scripts.crocker import *

betti_numbers = [0, 1]
#VANILLA CROCKER
#Which DataFrame columns to use as dimensions
DATA_COLS = ('x','y','angle')

#List of frame values to use, must be aligned for direct comparison
true_FRAME_LIST = range(20,120,1)
pred_FRAME_LIST = range(10,120,1) #starts at 10 because of angle computation
#compute the data for the crocker plot
#compute the data for the crocker plot
if 'angle' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
if 'vx' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
max_NUM_SAMPLE = int(1e4)
chosen_NUM_SAMPLE = int(1e4)

#Initial conditions
stage_idx = 37

sampled_idc = []
sampled_pars = []
sampled_losses = []
for sampled_idx in range(chosen_NUM_SAMPLE):
    sample_dir = './Simulated_Grid/ODE/stage_'+str(stage_idx)+'/sample_'+str(max_NUM_SAMPLE)+'/run_'+str(sampled_idx+1)+'/'
#    print(sampled_idx)
#    print(sample_dir)
    if os.path.isdir(sample_dir):
        if 'angle' in DATA_COLS:
            loss_path = sample_dir+'loss_angles.npy'
        if 'vx' in DATA_COLS:
            loss_path = sample_dir+'loss_velocities.npy'
        loss = np.load(loss_path)
        
        pars_path = sample_dir+'pars.npy'
        pars = np.load(pars_path)
                
        sampled_idc.append(sampled_idx+1)
        sampled_pars.append(pars)
        sampled_losses.append(loss)
#        print(loss)
        
save_results = {}
save_results['idc'] = sampled_idc
save_results['pars'] = sampled_pars
save_results['losses'] = sampled_losses
#print(save_results)
if 'angle' in DATA_COLS:
    np.save('Results/stage_'+str(stage_idx)+'/stage_'+str(stage_idx)+'_losses_angles.npy',save_results)
if 'vx' in DATA_COLS:
    np.save('Results/stage_'+str(stage_idx)+'/stage_'+str(stage_idx)+'_losses_velocities.npy',save_results)
