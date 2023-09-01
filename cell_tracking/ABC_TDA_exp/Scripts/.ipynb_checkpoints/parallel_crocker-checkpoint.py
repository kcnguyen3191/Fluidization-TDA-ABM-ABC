import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy
import concurrent.futures
from crocker import *

def run_save_crocker(args):
    
    C_idx, L_idx, DATA_COLS, FRAME_LIST, PROX_VEC = args
    DF_PATH = './Simulated_Grid_50T_100F/ODE/Cidx_'+str(C_idx+1).zfill(2)+'_Lidx_'+str(L_idx+1).zfill(2)+'/run_1/df.pkl'
    
    #read in the dataframe and filter positions
    filt_df = pd.read_pickle(DF_PATH)
    
    crocker = compute_crocker(filt_df,FRAME_LIST,PROX_VEC,
                              data_cols=DATA_COLS,betti=[0,1])
    
    SAVE_PATH = './Simulated_Grid_50T_100F/ODE/Cidx_'+str(C_idx+1).zfill(2)+'_Lidx_'+str(L_idx+1).zfill(2)+'/run_1/crocker.npy'
    np.save(SAVE_PATH,crocker)