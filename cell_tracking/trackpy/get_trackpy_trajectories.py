import os
import time
import pims
import trackpy as tp
from traj_utils import interpolate_traj,compute_velocity,change_units
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

'''
Driver script for processing all images in the pred_plots directory
'''

#Where to save trajectory plots and data
TRAJ_DIR = "../final_trajectories/"
if not os.path.isdir(TRAJ_DIR):
    os.makedirs(TRAJ_DIR)
#Whether or not to use tp.predict
NEAREST_VELO_PREDICT = False
#Minimum trajectory lenth to include in analysis
MIN_LENGTH = 10
#Whether or not to interpolate
INTERPOLATE = True
#Whether or not to remove drift
REMOVE_DRIFT = True
if REMOVE_DRIFT:
    DRIFT_DIR = os.path.join(TRAJ_DIR,'pixel_drift')
    if not os.path.isdir(DRIFT_DIR):
        os.mkdir(DRIFT_DIR)

for num in range(1,43):
    im_string = '/home/mars/external4/scratch_assay_final_pmaps/20_overlap/scratch3t3_1nMPDGF1_w2Cy5_s'+str(num)+'_t*.png'
    beg = time.time()
    frames = pims.ImageSequence(im_string)

    # f is a pandas dataframe that has cell locations over all frames
    tp.quiet()
    f = tp.batch(frames,11);

    #t is a new dataframe that contains the connections between the particles
    # from frame f, creating trajectories for cells.
    if not NEAREST_VELO_PREDICT:
        t = tp.link(f,search_range=10,memory=5)
    else:
        pred = tp.predict.NearestVelocityPredict()
        t = pred.link_df(f,search_range=10,memory=5)

    #clip trajectories shorter than MIN_LENGTH frames
    if MIN_LENGTH > 0:
        t = tp.filter_stubs(t,MIN_LENGTH)

    #Interpolate missing frames for each remaining particle
    if INTERPOLATE:
        t = interpolate_traj(t)

    #Adjust for jitter by subtracting the computed image drift
    if REMOVE_DRIFT:
        drift = tp.compute_drift(t)#compute overall drift
        drift.x = drift.x.astype(int)
        drift.y = drift.y.astype(int)
        drift_path = os.path.join(DRIFT_DIR,'drift_pixels_'+str(num)+'.pkl')
        drift.to_pickle(drift_path)#save drift for plotting purposes
        t = tp.subtract_drift(t.copy(),drift)#subtract drift from trajectories

    #Change to biologically-relevant units
    t = change_units(t)

    #Correct for stage 9's missing frame
    if num == 9:
        t.loc[t['frame']>89,['frame']] = t.loc[t['frame']>89,['frame']]+1
        t.loc[t['frame']>89,['t']] = t.loc[t['frame']>89,['t']]+1/6

    #Compute velocities in correct units
    t = compute_velocity(t)

    #Compute angles
    t['angle'] = np.arctan2(t['vy'],t['vx'])

    save_path = os.path.join(TRAJ_DIR,"stage_"+str(num)+".pkl")
    t.to_pickle(save_path)
    plt.figure(figsize=(10,7))
    tp.plot_traj(t)
    plt.savefig(os.path.join(TRAJ_DIR,"stage_"+str(num)+"_traj_plot.png"))
    plt.close()
    print(time.time()-beg,"seconds for stage",num)
