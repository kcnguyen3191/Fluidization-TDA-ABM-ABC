import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

'''
Functions for processing trajectory DataFrames by interpolating,
filtering, and computing velocity.
'''

def filter_dist(df,pixel_dist):
    '''Filters out trajectories that travel fewer than pixel_dist pixels over
    their entire lifespan using displacements from compute_velocity vectors.
    '''
    particles = np.sort(df.particle.unique())
    dists = [np.linalg.norm(df.loc[df['particle']==i][['vx','vy']].to_numpy(),
                            ord=2,axis=1).sum() for i in particles]
    keep = particles[[d > pixel_dist for d in dists]]
    return df[df.particle.isin(keep)]

def interpolate_traj(df):
    '''Performs linear interpolation on "df" to get a position at each time
    step for each particle.
    '''
    new_df = pd.DataFrame(np.zeros((1,4),dtype=np.float64),
                          columns=['x','y','particle','frame'])

    for p in df.particle.unique():
        particle = df[df['particle']==p]
        #Exclude single-frame tracks
        if particle.shape[0] <= 1:
            continue

        #Get relevant columns
        t = np.array(particle['frame'])
        x = np.array(particle['x'])
        y = np.array(particle['y'])
        #Get all frames between cell's first and last appearance
        all_frames = np.arange(particle['frame'].iloc[0],
                               particle['frame'].iloc[-1]+1)
        #Use linear interpolation on x and y to fill in trajectories
        fx = interp1d(t,x)
        fy = interp1d(t,y)
        new_x = fx(all_frames)
        new_y = fy(all_frames)
        #Concatenate to new_df
        particle_df = pd.DataFrame({'x':new_x,'y':new_y,
                                    'particle':p,'frame':all_frames})
        new_df = pd.concat((new_df,particle_df),axis=0)
        new_df['particle'] = new_df['particle'].astype(np.uint16)
        new_df['frame'] = new_df['frame'].astype(np.uint16)

    #Reset indices and get rid of zeros row
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.drop(0)
    new_df = new_df.reset_index(drop=True)

    return new_df

def change_units(df,
                 dim_conv=(1.157*1000),time_conv=6):
    '''Converts from pixels and frames to millimeters and hours.'''
    df['x'] = df['x'].apply(lambda x: x/dim_conv)
    df['y'] = df['y'].apply(lambda y: y/dim_conv)
    df['t'] = df['frame'].apply(lambda t: t/time_conv)
    return df

def compute_velocity(df):
    '''Uses numpy.gradient to compute velocity vector at each time step for
    each particle.
    '''
    new_df = pd.DataFrame(np.zeros((1,7),dtype=np.float64),
                          columns=['t','x','y','vx','vy','particle','frame'])

    for p in df.particle.unique():
        particle = df[df['particle']==p]
        #Exclude particles that have too few frames
        if particle.shape[0] <= 2:
            continue

        #Use numpy to compute velocity
        x = np.array(particle['x'])
        y = np.array(particle['y'])
        t = np.array(particle['t'])
        vx = np.gradient(x,t)
        vy = np.gradient(y,t)
        #Concatenate to new_df
        particle_df = pd.DataFrame({'t':t,'x':x,'y':y,'vx':vx,'vy':vy,'particle':p,
                                    'frame':particle['frame'].reset_index(drop=True)})
        new_df = pd.concat((new_df,particle_df),axis=0)
        new_df['particle'] = new_df['particle'].astype(np.uint16)
        new_df['frame'] = new_df['frame'].astype(np.uint16)

    #Reset indices and get rid of zeros row
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.drop(0)
    new_df = new_df.reset_index(drop=True)

    return new_df
