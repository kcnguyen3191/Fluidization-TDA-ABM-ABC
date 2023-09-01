import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DF_PATH = './final_trajectorie/stage_41.pkl'
IC_FRAME = 6
FILTER_DIST = 0.04
BINS = np.linspace(-2*FILTER_DIST,2*FILTER_DIST,201)
SAVE_DIR = './sigma_stationary/'
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

#Read in dataframe
traj_df = pd.read_pickle(DF_PATH)
ic_df = traj_df.loc[traj_df['frame']==IC_FRAME,['x','y','t','particle']]

#Make new dataframe with deltas from IC_FRAME
new_df = traj_df[['t','x','y','particle','frame']].merge(
            ic_df[['t','x','y','particle']],
            on='particle',suffixes=(None,'_initial'))
new_df['dx'] = new_df['x'] - new_df['x_initial']
new_df['dy'] = new_df['y'] - new_df['y_initial']

#Filter on particles with less than FILTER_DIST absolute distance in either
#direction between initial position and final position
stationary_particles = new_df.loc[((new_df['frame']==max(new_df['frame'])) &
                                   (new_df['dx'].abs() <= FILTER_DIST) &
                                   (new_df['dy'].abs() <= FILTER_DIST)),
                                  'particle'].unique()
new_df = new_df.loc[new_df['particle'].isin(stationary_particles)]

#For each frame, get sigmas for x and y
frame_vec = [f for f in range(IC_FRAME+1,new_df.frame.max()+1) if f!=90]
x_sigmas,y_sigmas = [],[]
covs = []
for f in frame_vec:
    df = new_df.loc[new_df['frame']==f].reset_index(drop=True)
    #Plot delta distributions
    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True)
    ax1.hist(df['dx'],bins=BINS)
    ax2.hist(df['dy'],bins=BINS)
    ax1.set_title(r'$\Delta x$ distribution')
    ax2.set_title(r'$\Delta y$ distribution')
    ax1.set_ylim((0,10))
    fig.suptitle('Frame {0}'.format(f))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,'frame_{0}_dists.png'.format(f)))
    plt.close()

    #Get sigmas
    dt = df['t'][0]-df['t_initial'][0]
    print(dt)
    cov_mat = np.cov(np.stack([df['dx'],df['dy']],axis=0))
    x_sigmas.append(np.sqrt(cov_mat[0,0]/dt))
    y_sigmas.append(np.sqrt(cov_mat[1,1]/dt))
    covs.append(cov_mat[0,1])

plt.hist(covs)
plt.title('Histogram of position delta covariances')
plt.tight_layout()
plt.show()
plt.close()

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True)
ax1.plot(frame_vec,x_sigmas,'x-k')
ax2.plot(frame_vec,y_sigmas,'x-k')
ax1.set_title(r'$\sigma_x$ approximations')
ax2.set_title(r'$\sigma_y$ approximations')
plt.tight_layout()
plt.show()
