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
    
    iSample, NUM_SAMPLE, DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, PROX_VEC = args
    DF_PATH = './Simulated_Grid/ODE/sample_'+str(NUM_SAMPLE)+'/run_'+str(iSample+1)+'/df.pkl'
    unscale_num = 25
#    if os.path.isfile(DF_PATH):
    #read in the dataframe and filter positions
    filt_df = pd.read_pickle(DF_PATH)
    
    if 'angle' in DATA_COLS:
        filt_df, _ = filtering_df(filt_df, pred_FRAME_LIST, track_len=10, max_frame=500, min_speed=0)
    filt_df.x = filt_df.x/unscale_num
    filt_df.y = filt_df.y/unscale_num
    filt_df.vx = filt_df.vx/unscale_num
    filt_df.vy = filt_df.vy/unscale_num
#    print(max(filt_df.x))
    crocker = compute_crocker_custom(filt_df,true_FRAME_LIST,PROX_VEC,
                              data_cols=DATA_COLS,betti=[0,1])
    
    if 'vx' in DATA_COLS:
        SAVE_PATH = './Simulated_Grid/ODE/sample_'+str(NUM_SAMPLE)+'/run_'+str(iSample+1)+'/crocker_velocities.npy'
    if 'angle' in DATA_COLS:
        SAVE_PATH = './Simulated_Grid/ODE/sample_'+str(NUM_SAMPLE)+'/run_'+str(iSample+1)+'/crocker_angles.npy'
    np.save(SAVE_PATH,crocker)

#VANILLA CROCKER
#Which DataFrame columns to use as dimensions
DATA_COLS = ('x','y','angle')

#List of frame values to use, must be aligned for direct comparison
true_FRAME_LIST = range(20,120,1)
pred_FRAME_LIST = range(10,120,1) #starts at 10 because of angle computation
#compute the data for the crocker plot
if 'angle' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
if 'vx' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker

#Number of datasets to make
NUM_SAMPLE = 10000


list_tuples = []
for iSample in range(NUM_SAMPLE):
    list_tuples.append((iSample, NUM_SAMPLE, DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, PROX_VEC))

#run_save_crocker(list_tuples[0])
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(run_save_crocker, list_tuples)
