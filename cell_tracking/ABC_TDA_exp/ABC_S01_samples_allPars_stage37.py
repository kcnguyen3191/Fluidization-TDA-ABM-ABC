import numpy as np
from itertools import permutations
from math import factorial
import scipy.optimize as opt
from functools import partial
import time
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os
import glob
import imageio as io
from itertools import repeat
from ripser import ripser
import scipy
import matplotlib as mpl
from functools import partial
import concurrent.futures
from Scripts.Dorsogna_fluidization import *

def run_simulation(pars, ic_vec, time_vec, num_sample, iSample, stage_idx):
    SIGMA, ALPHA, BETA, CA, CR, LA, LR = pars
    T0 = np.min(time_vec)
    TF = np.max(time_vec)
    DT = time_vec[1] - time_vec[0]

    par_dir = 'stage_'+str(stage_idx)+'/sample_'+str(num_sample)
    #Where to save the runs
    if SIGMA == 0:
        FIGURE_PATH = './Simulated_Grid/ODE/'+par_dir+'/'
    elif SIGMA > 0:
        FIGURE_PATH = './Simulated_Grid/SDE/'+par_dir+'/'
    else:
        raise ValueError("{0} is an invalid value for SIGMA".format(SIGMA))

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)
    
    save_dir = os.path.join(FIGURE_PATH,'run_{0}'.format(iSample+1))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    pickle_path = os.path.join(save_dir,'df.pkl')
    if not os.path.isfile(pickle_path):
        
        pars_path = os.path.join(save_dir,'pars.npy')
        np.save(pars_path,pars)
        
        #Simulate using appropriate integrator
        MODEL_CLASS = Dorsogna_fluidization
        model = MODEL_CLASS(sigma=SIGMA,alpha=ALPHA,beta=BETA,cA=CA,cR=CR,lA=LA,lR=LR)
        if SIGMA == 0:
            model.ode_rk4(ic_vec,T0,TF,DT)
        elif SIGMA > 0:
            model.sde_maruyama(ic_vec,T0,TF,return_time=DT)
        else:
            raise ValueError("{0} is an invalid value for SIGMA".format(SIGMA))
        
        #Save results as dataframe
        results = model.results_to_df(time_vec)
        results.to_pickle(pickle_path)
#         print("Saved to",pickle_path)
        
        #Plot gif of simulated positions
        # model.position_gif(save_dir,time_vec)

def simulation_wrapper(args):
    
    SIGMA, ALPHA, BETA, CA, CR, LA, LR, ic_vec, time_vec, num_sample, iSample, stage_idx = args
    pars = [SIGMA, ALPHA, BETA, CA, CR, LA, LR]
    run_simulation(pars, ic_vec, time_vec, num_sample, iSample, stage_idx)

###ARGS
#What time to use as initial
T0 = 1
#What time to end the simulation
TF = 21
#How often to make a new frame of data
DT = 1/6
#Make time vector
time_vec = np.arange(T0,TF+DT,DT)

#Number of datasets to make
NUM_SAMPLE = int(1e4)

#Initial conditions
stage_idx = 37
ic_vec = np.load('IC_data/subsampled_data/stage_'+str(stage_idx)+'_subsampled_IC.npy')

#Stochastic diffusivity parameter
SIGMA = 0 #0.05
alpha_LB = 0.001
alpha_UB = 0.1
v2 = 0.06773**2
beta_LB  = 0.1
beta_UB  = 1.0
cA_LB    = 0.0001
cA_UB    = 0.006
cR_LB    = 0.0005
cR_UB    = 0.008
lA_LB    = 0.005
lA_UB    = 0.03
lR_LB    = 0.001
lR_UB    = 0.02

c_LB = 0.1
c_UB = 3.0
l_LB = 0.1
l_UB = 3.0

list_tuples = []
for iSample in range(NUM_SAMPLE):
    BETA  = 0.5
    ALPHA = 1.0
    
    # Uniformly sampling C, L
    C     = np.random.uniform(low=c_LB,high=c_UB)
    CA    = 1
    CR    = CA*C
    
    L     = np.random.uniform(low=l_LB,high=l_UB)
    LA    = 1
    LR    = LA*L

    list_tuples.append((SIGMA, ALPHA, BETA, CA, CR, LA, LR, ic_vec, time_vec, NUM_SAMPLE, iSample, stage_idx))
    
#simulation_wrapper(list_tuples[0])
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(simulation_wrapper, list_tuples)
