import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy
import concurrent.futures
from Scripts.crocker import *

def filtering_df(filt_df, FRAME_LIST, track_len=10, max_frame=128, min_speed=None):
    
    if min_speed is None:
        #make a histogram of velocities (displacement over some number of frames) of all cells at all times
        object = filt_df.copy()
        mags_xy = []
        for idx in np.arange(track_len,max_frame):
            p_t1=object.particle.values[object.frame==idx-track_len]
            p_t2=object.particle.values[object.frame==idx]

            object2= object[(object["particle"].isin(p_t1)) & (object["particle"].isin(p_t2))]

            frame_t1 = object2[object2['frame']==idx-track_len]
            frame_t2 = object2[object2['frame']==idx]
            delta_x = frame_t2.x.values-frame_t1.x.values
            delta_y = frame_t2.y.values-frame_t1.y.values

            mag_xy = np.sqrt(np.square(delta_x)+np.square(delta_y))
            mags_xy.append(mag_xy)

        mags_xy_data=np.concatenate(mags_xy,axis=0)
        a1,b1,c1=scipy.stats.lognorm.fit(mags_xy_data)
        rv = scipy.stats.lognorm(a1,b1,c1)
        min_speed=rv.ppf(.25)

    ##NEW
    #get fast moving cells, above 25th percentile for displacement over 10 frames
    #min_speed = 0.00656070884994189 #calucated in #figures_for_paper.ipynb
    #go through frames 10 to 126 and add particles to a new data frame
    #only keep those that move fast enough
    #also calculate angle of displacement
    #keep a data frame with x,y,angle,magnitude, and frame
    new_data = pd.DataFrame(columns=['x','y','vx','vy','angle','mag','frame'])

    num_cells_in_frame = []
    num_cells_in_frame_all = []

    for frame_idx in FRAME_LIST:
        p_t1=filt_df.particle.values[filt_df.frame==frame_idx-track_len]
        p_t2=filt_df.particle.values[filt_df.frame==frame_idx]
        object2= filt_df[(filt_df["particle"].isin(p_t1)) & (filt_df["particle"].isin(p_t2))]
        frame_t2 = object2[object2['frame']==frame_idx]
        frame_t1 = object2[object2['frame']==frame_idx-track_len]
        delta_x = frame_t2.x.values-frame_t1.x.values
        delta_y = frame_t2.y.values-frame_t1.y.values
        angle_xy = np.arctan2(delta_y,delta_x)
        deg_xy = np.mod(np.degrees(angle_xy),360)
        mag_xy = np.sqrt(np.square(delta_x)+np.square(delta_y))
        x = frame_t2.x.values
        y = frame_t2.y.values
        vx = frame_t2.vx.values
        vy = frame_t2.vy.values
        particle = frame_t2.particle.values
        count=0
        all_count = 0
        for idx,val in enumerate(x):
            all_count +=1
            if mag_xy[idx]>min_speed:
                count+=1
                df1 = pd.DataFrame({'x':[x[idx]],'y':[y[idx]],'vx':[vx[idx]],'vy':[vy[idx]],'angle':[deg_xy[idx]],'mag':[mag_xy[idx]],'frame':[frame_idx],'particle':[particle[idx]]})
                new_data=pd.concat([new_data,df1],ignore_index=True,axis=0)
        num_cells_in_frame.append(count)
        num_cells_in_frame_all.append(len(p_t2))
        
    return new_data, min_speed


def run_compute_crocker_angle(args):
    
    sampled_idx, stage_idx, DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, betti_numbers, PROX_VEC, max_NUM_SAMPLE, chosen_NUM_SAMPLE = args
    unscale_num = 25
    sampled_path = './Simulated_Grid/ODE/stage_'+str(stage_idx)+'/sample_'+str(max_NUM_SAMPLE)+'/run_'+str(sampled_idx+1)+'/df.pkl'
    sampled_df = pd.read_pickle(sampled_path)
    
    if 'angle' in DATA_COLS:
        sampled_df, _ = filtering_df(sampled_df, pred_FRAME_LIST, track_len=10, max_frame=126, min_speed=0)
    sampled_df.x = sampled_df.x/unscale_num
    sampled_df.y = sampled_df.y/unscale_num
    sampled_df.vx = sampled_df.vx/unscale_num
    sampled_df.vy = sampled_df.vy/unscale_num
    sampled_crocker = compute_crocker_custom(sampled_df, true_FRAME_LIST, PROX_VEC, data_cols=DATA_COLS,betti=betti_numbers)
    
    if 'vx' in DATA_COLS:
        save_path = './Simulated_Grid/ODE/stage_'+str(stage_idx)+'/sample_'+str(max_NUM_SAMPLE)+'/run_'+str(sampled_idx+1)+'/crocker_velocities.npy'
    if 'angle' in DATA_COLS:
        save_path = './Simulated_Grid/ODE/stage_'+str(stage_idx)+'/sample_'+str(max_NUM_SAMPLE)+'/run_'+str(sampled_idx+1)+'/crocker_angles.npy'
    
    np.save(save_path,sampled_crocker)
    
betti_numbers = [0, 1]
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
    
max_NUM_SAMPLE = int(1e4)
chosen_NUM_SAMPLE = int(1e4)

#Initial conditions
stage_idx = 37


list_tuples = []
list_tuples = []
for idx in range(chosen_NUM_SAMPLE):
    list_tuples.append((idx, stage_idx, DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, betti_numbers, PROX_VEC, max_NUM_SAMPLE, chosen_NUM_SAMPLE))

#run_compute_crocker_angle(list_tuples[0])
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(run_compute_crocker_angle, list_tuples)
