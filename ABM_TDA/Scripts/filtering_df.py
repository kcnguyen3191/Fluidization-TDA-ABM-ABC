import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy
import concurrent.futures

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
