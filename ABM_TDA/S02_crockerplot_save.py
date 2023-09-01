import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy
import concurrent.futures
from Scripts.filtering_df import *
from Scripts.crocker import *

def run_save_crocker(args):
    
    C_idx, L_idx, DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, PROX_VEC = args
    DF_PATH = './Simulated_Grid/ODE/Cidx_'+str(C_idx+1).zfill(2)+'_Lidx_'+str(L_idx+1).zfill(2)+'/run_1/df.pkl'
    
    #read in the dataframe and filter positions
    filt_df = pd.read_pickle(DF_PATH)
    if 'angle' in DATA_COLS:
        filt_df, _ = filtering_df(filt_df, pred_FRAME_LIST, track_len=10, max_frame=128, min_speed=0)
    filt_df.x = filt_df.x/25
    filt_df.y = filt_df.y/25 #0.4, ~
    filt_df.vx = filt_df.vx/25
    filt_df.vy = filt_df.vy/25
    crocker = compute_crocker_custom(filt_df,true_FRAME_LIST,PROX_VEC,
                              data_cols=DATA_COLS,betti=[0,1])
    if 'vx' in DATA_COLS:
        SAVE_PATH = './Simulated_Grid/ODE/Cidx_'+str(C_idx+1).zfill(2)+'_Lidx_'+str(L_idx+1).zfill(2)+'/run_1/crocker_velocities.npy'
    if 'angle' in DATA_COLS:
        SAVE_PATH = './Simulated_Grid/ODE/Cidx_'+str(C_idx+1).zfill(2)+'_Lidx_'+str(L_idx+1).zfill(2)+'/run_1/crocker_angles.npy'
    np.save(SAVE_PATH,crocker)

#VANILLA CROCKER
#Which DataFrame columns to use as dimensions
DATA_COLS = ('x','y','angle')
# Frame 10: get initial conditions (x,y,vx,vy)
# Frame 20:
#List of frame values to use, must be aligned for direct comparison
true_FRAME_LIST = range(20,120,1)
pred_FRAME_LIST = range(10,120,1) #starts at 10 because of angle computation
#compute the data for the crocker plot
if 'angle' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
if 'vx' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker


Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)

list_tuples = []
for C_idx, C in enumerate(Cs):
    for L_idx, L in enumerate(Ls):
        list_tuples.append((C_idx, L_idx, DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, PROX_VEC))

NUM_RUNS = 1

for r in range(NUM_RUNS):
    # For debugging
#    run_save_crocker(list_tuples[0])

    # Parallel computing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(run_save_crocker, list_tuples)
