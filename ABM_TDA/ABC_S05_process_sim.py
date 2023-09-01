import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy

import concurrent.futures
from scipy.integrate import ode
import glob
import imageio as io
from itertools import repeat

from Scripts.DorsognaNondim import *
from Scripts.crocker import *

def run_simulation(pars, ic_vec, time_vec, iRUN, opt_alg=None):
    SIGMA, ALPHA, BETA, C_idx, C, L_idx, L = pars
    T0 = np.min(time_vec)
    TF = np.max(time_vec)
    DT = time_vec[1] - time_vec[0]

    par_dir = 'Cidx_'+str(C_idx).zfill(2)+'_Lidx_'+str(L_idx).zfill(2)
    #Where to save the runs
    if SIGMA == 0:
        FIGURE_PATH = './Simulated_Grid/ODE/'+par_dir+'/'
    elif SIGMA > 0:
        FIGURE_PATH = './Simulated_Grid/SDE/'+par_dir+'/'
    else:
        raise ValueError("{0} is an invalid value for SIGMA".format(SIGMA))

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)
    
    if opt_alg is not None:
        if opt_alg == "NM":
            save_dir = os.path.join(FIGURE_PATH,'run_{0}/'.format(iRUN+1))
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            pickle_path = os.path.join(save_dir,'df_NM.pkl')
        elif opt_alg == "ABC":
            save_dir = os.path.join(FIGURE_PATH,'run_{0}/'.format(iRUN+1))
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            pickle_path = os.path.join(save_dir,'df_ABC.pkl')

    #Simulate using appropriate integrator
    MODEL_CLASS = DorsognaNondim
    model = MODEL_CLASS(sigma=SIGMA,alpha=ALPHA,beta=BETA,
                       c=C,l=L)
    if SIGMA == 0:
        model.ode_rk4(ic_vec,T0,TF,DT)
    elif SIGMA > 0:
        model.sde_maruyama(ic_vec,T0,TF,return_time=DT)
    else:
        raise ValueError("{0} is an invalid value for SIGMA".format(SIGMA))

    #Save results as dataframe
    results = model.results_to_df(time_vec)
    results.to_pickle(pickle_path)
    
    #Plot gif of simulated positions
    model.position_gif(save_dir,time_vec)

Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)

pars_idc = [(18,4),(7,25),(20,1),(9,6),(5,5),(2,15),(15,7),(25,25),(20,15)]

#What time to use as initial
T0 = 1
#What time to end the simulation
TF = 21
#How often to make a new frame of data
DT = 1/6
#Make time vector
time_vec = np.arange(T0,TF+DT,DT)
#Initial conditions
rng = np.random.default_rng()

num_agents = 300
#Number of datasets to make
NUM_RUNS = 100

ic_vec = np.load('ic_vec.npy',allow_pickle=True)

#Stochastic diffusivity parameter
SIGMA = 0 #0.05
#alpha
ALPHA = 1.0
BETA = 0.5
iRUN = 0

for pars_idx in pars_idc:
    Cidx, Lidx = pars_idx
    C_true = Cs[Cidx-1]
    L_true = Ls[Lidx-1]
    
    # Get ABC results:
    loss_path = './Simulated_Grid/ODE/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'/run_1/sample_losses_angles.npy'
    sample_losses = np.load(loss_path,allow_pickle=True).item()
    
    sample_idx = []
    C_vals = []
    L_vals = []
    losses = []
    
    # max_Sample = len(sample_losses)
    samples_idcs = list(sample_losses)
    for iSample in samples_idcs:
        iSample = int(iSample)
        sample_idx.append(iSample)
        losses.append(sample_losses[str(iSample)]['loss'])
        C_vals.append(sample_losses[str(iSample)]['sampled_pars'][3])
        L_vals.append(sample_losses[str(iSample)]['sampled_pars'][4])
    losses = np.array(losses)
    C_vals = np.array(C_vals)
    L_vals = np.array(L_vals)

    nan_idc = np.argwhere(np.isnan(losses))

    losses = np.delete(losses, nan_idc, axis=0)
    C_vals = np.delete(C_vals, nan_idc, axis=0)
    L_vals = np.delete(L_vals, nan_idc, axis=0)
    
    min_loss_idx = np.argmin(losses)
    C_min = C_vals[min_loss_idx]
    L_min = L_vals[min_loss_idx]

    distance_threshold = np.percentile(losses, 1, axis=0)
    
    # Plot NM and ABC results
    belowTH_idc = np.where(losses < distance_threshold)[0]
    C_median = np.median(C_vals[belowTH_idc])
    L_median = np.median(L_vals[belowTH_idc])
    
    pars = [SIGMA, ALPHA, BETA, Cidx, C_median, Lidx, L_median]
    
    # Run Nelder-Mead result simulation
    run_simulation(pars, ic_vec, time_vec, iRUN, opt_alg="ABC")
