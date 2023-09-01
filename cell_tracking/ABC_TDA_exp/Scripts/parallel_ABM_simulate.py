import numpy as np
import pandas as pd
import concurrent.futures
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os
import glob
import imageio as io
from itertools import repeat
from DorsognaNondim import *
from Dorsogna import *

def run_simulation(pars, ic_vec, time_vec, iRUN):
    SIGMA, ALPHA, BETA, C_idx, C, L_idx, L = pars
    T0 = np.min(time_vec)
    TF = np.max(time_vec)
    DT = time_vec[1] - time_vec[0]

    par_dir = 'Cidx_'+str(C_idx+1).zfill(2)+'_Lidx_'+str(L_idx+1).zfill(2)
    #Where to save the runs
    if SIGMA == 0:
        FIGURE_PATH = './Simulated_Grid_50T_100F/ODE/'+par_dir+'/'
    elif SIGMA > 0:
        FIGURE_PATH = './Simulated_Grid_50T_100F/SDE/'+par_dir+'/'
    else:
        raise ValueError("{0} is an invalid value for SIGMA".format(SIGMA))

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)
    
    save_dir = os.path.join(FIGURE_PATH,'run_{0}'.format(iRUN+1))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    pickle_path = os.path.join(save_dir,'df.pkl')
    if not os.path.isfile(pickle_path):
    
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
#         print("Saved to",pickle_path)
        
        #Plot gif of simulated positions
        model.position_gif(save_dir,time_vec)

def simulation_wrapper(args):

    sim_values, iRUN = args
    SIGMA, ALPHA, BETA, C_idx, C_val, L_idx, L_val, ic_vec, time_vec = sim_values
    pars = [SIGMA, ALPHA, BETA, C_idx, C_val, L_idx, L_val]
    
    run_simulation(pars, ic_vec, time_vec, iRUN)