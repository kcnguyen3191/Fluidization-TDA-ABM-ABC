import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import glob
import imageio as io
import scipy
import matplotlib as mpl
from Scripts.Dorsogna_fluidization import *
from Scripts.crocker import *
import warnings
warnings.filterwarnings("ignore")

def compute_angle_df(filt_df,FRAME_LIST,track_len):
    
    #make a histogram of velocities (displacement over some number of frames) of all cells at all times
    object = filt_df.copy()
    mags_xy = []
    track_len=10
    for idx in np.arange(track_len,128):
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
    x=np.linspace(0,0.035,100)
#     plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
#     n, bins, patches = plt.hist(mags_xy_data, 50, density=True, facecolor='g', alpha=0.75)
    #min_speed=rv.ppf(.25)
    min_speed=0

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
                #print(df1)
                new_data=pd.concat([new_data,df1],ignore_index=True,axis=0)
                #sc=ax.scatter(x[idx],y[idx],c=deg_xy[idx],vmin=0,vmax=360,cmap='hsv',marker=marker, s=(markersize*scale)**2)
        num_cells_in_frame.append(count)
        num_cells_in_frame_all.append(len(p_t2))
        
    return new_data

def run_simulation(pars, ic_vec, time_vec, stage_idx):
    SIGMA, ALPHA, BETA, CA, CR, LA, LR = pars
    T0 = np.min(time_vec)
    TF = np.max(time_vec)
    DT = time_vec[1] - time_vec[0]

    par_dir = 'stage_'+str(stage_idx)
    #Where to save the runs
    if SIGMA == 0:
        FIGURE_PATH = './Results/'+par_dir+'/'
    elif SIGMA > 0:
        FIGURE_PATH = './Results/'+par_dir+'/'
    else:
        raise ValueError("{0} is an invalid value for SIGMA".format(SIGMA))

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)
    
    save_dir = FIGURE_PATH
    
    pickle_path = os.path.join(save_dir,'df.pkl')
        
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
    # results.to_pickle(pickle_path)

    #Plot gif of simulated positions
    model.position_gif(save_dir,time_vec)
    
    return results

stage_idx = 37

results = np.load('Results/stage_'+str(stage_idx)+'/stage_'+str(stage_idx)+'_losses_angles.npy',allow_pickle=True).item()

losses = np.asarray(results['losses'])
pars =  np.asarray(results['pars'])
idc =  np.asarray(results['idc'])

nan_idc = np.argwhere(np.isnan(losses))

losses = np.delete(losses, nan_idc, axis=0)
pars = np.delete(pars, nan_idc, axis=0)
idc = np.delete(idc, nan_idc, axis=0)
p01, p05, p10 = np.percentile(losses, [1, 5, 10], axis=0)##[0.1, 100], axis=0)

distance_thresholds = [p01, p05, p10]#, p25]#, p50]#[2500]

betti_numbers = [0, 1]
#VANILLA CROCKER
#Which DataFrame columns to use as dimensions
DATA_COLS = ('x','y','angle')
###ARGS
#What time to use as initial
T0 = 1
#What time to end the simulation
TF = 21
#How often to make a new frame of data
DT = 1/6
#Make time vector
time_vec = np.arange(T0,TF+DT,DT)
FRAME_LIST = range(10,126,1) #starts at 10 because of angle computation

#Number of datasets to make
NUM_SAMPLE = int(1e4)
#Initial conditions
ic_vec = np.load('IC_data/subsampled_data/stage_'+str(stage_idx)+'_subsampled_IC.npy')
# ic_vec[0:600] = ic_vec[0:600]*100
ic_vec = ic_vec

Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)
C_mesh, L_mesh = np.meshgrid(Cs, Ls, indexing='ij')
count = 0
for distance_threshold in distance_thresholds:
    belowTH_idc = np.where(losses < distance_threshold)[0]
    SIGMA_median = np.median(pars[belowTH_idc,0])
    BETA_median  = np.median(pars[belowTH_idc,2])
    ALPHA_median = np.median(pars[belowTH_idc,1])
    CA_median    = np.median(pars[belowTH_idc,3])
    CR_median    = np.median(pars[belowTH_idc,4])
    LA_median    = np.median(pars[belowTH_idc,5])
    LR_median    = np.median(pars[belowTH_idc,6])
    
    C_median = CR_median/CA_median
    L_median = LR_median/LA_median
    
    sim_pars = [SIGMA_median, ALPHA_median, BETA_median, CA_median, CR_median, LA_median, LR_median]
    median_df = run_simulation(sim_pars, ic_vec, time_vec, stage_idx)
    median_df = compute_angle_df(median_df,FRAME_LIST,track_len=10)
    
    if count == 0:
        pickle_path = './Results/'+'stage_'+str(stage_idx)+'/ABC_df_angles_p01_angles.pkl'
    if count == 1:
        pickle_path = './Results/'+'stage_'+str(stage_idx)+'/ABC_df_angles_p05_angles.pkl'
    if count == 2:
        pickle_path = './Results/'+'stage_'+str(stage_idx)+'/ABC_df_angles_p10_angles.pkl'
        
    median_df.to_pickle(pickle_path)
    
    
    sample_count_map = np.zeros((len(Cs),len(Ls)))
    for belowTH_idx in belowTH_idc:
        SIGMA, ALPHA, BETA, CA, CR, LA, LR = pars[belowTH_idx]
        C_val = CR/CA
        L_val = LR/LA
        
        Cidx_belowTH = np.argmin(np.abs(Cs-C_val))
        Lidx_belowTH = np.argmin(np.abs(Ls-L_val))
        sample_count_map[Cidx_belowTH,Lidx_belowTH] += 1
    sample_count_map = sample_count_map/NUM_SAMPLE
    fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=400)
    ct = plt.contourf(C_mesh, L_mesh, sample_count_map)
    plt.scatter(C_median,L_median,
                c="k",
                linewidths = 0.5,
                marker = 'o',
                edgecolor = "k",
                s = 100,
                label='ABC-Median')
    cbar = plt.colorbar(ct)
    ax.legend(fontsize=15)
    ax.set_aspect('equal', adjustable='box')
    
    plt.xlabel("C", fontsize=20)
    plt.ylabel("L", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xticks([0.1, 1.0, 2.0, 3.0])
    ax.set_yticks([0.1, 1.0, 2.0, 3.0])
    
    fig.tight_layout()
    plt.title('Stage 37', fontsize=18)
    if count == 0:
        fig.savefig('./Results/stage_'+str(stage_idx)+'/ABC_posterior_p01_angles.pdf',bbox_inches='tight')
    if count == 1:
        fig.savefig('./Results/stage_'+str(stage_idx)+'/ABC_posterior_p05_angles.pdf',bbox_inches='tight')
    if count == 2:
        fig.savefig('./Results/stage_'+str(stage_idx)+'/ABC_posterior_p10_angles.pdf',bbox_inches='tight')
        
    count += 1
