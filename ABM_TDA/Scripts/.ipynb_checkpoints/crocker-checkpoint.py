import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy
import concurrent.futures

def compute_crocker(traj_df,frame_vec,prox_vec,
                    data_cols=('x','y','vx','vy'),
                    betti=[0,1]):
    '''Compute crockers for specific Betti numbers given a trajectory dataframe.

    Inputs:
        traj_df (DataFrame): dataframe with ['x','y','vx','vy','frame']
        frame_vec (list): list of frames to sample for the crocker
        prox_vec (list): list of prox values to use for the crocker

    Kwargs:
        data_cols (tuple): which df columns to use in the point cloud
        betti (list of ints): which Betti numbers to yield
    
    Output:
        crocker (ndarray): crocker (len(time_vec),len(prox_vec),len(betti))
    '''
    betti_curves = []
    for j in frame_vec:#for each desired frame
        #Get relevant data from df
        data = traj_df[traj_df['frame']==j]
        data = data[list(data_cols)].to_numpy()
        print("\r{0} points in frame {1}".format(data.shape[0],j),end='')
        #Compute barcodes
        barcodes = ripser(data,maxdim=max(betti))['dgms']
        #Replace inf with maximum H1 death value
        barcodes[0][barcodes[0] == np.inf] = max(barcodes[1].max(),prox_vec.max())
        #Compute Betti numbers
        betti_curves.append(compute_betti(barcodes,prox_vec)[:,betti])

    #print("\n",end='')
    crocker = np.stack(betti_curves,axis=0)
    return crocker

def compute_betti(barcodes,prox_vec):
    '''Computes the Betti curve of barcodes given a list of proximity values.
    '''
    betti_stack = [
        [np.sum(np.logical_and(b[:,0]<prox,b[:,1]>=prox)) for prox in prox_vec]
                   for b in barcodes]
    return np.stack(betti_stack,axis=1)

def plot_crocker(crocker,prox_vec,
                 save_path=None):
    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(7.77,4.8))
    bcs0 = ax1.contourf(*np.meshgrid(range(1,crocker.shape[0]+1),
                                     prox_vec,indexing='ij'),
                        crocker[:,:,0],(1,50,100,150,200,250))
    fig.colorbar(bcs0,ax=ax1)
    ax1.set_title("Betti 0")
    bcs1 = ax2.contourf(*np.meshgrid(range(1,crocker.shape[0]+1),
                                     prox_vec,indexing='ij'),
                        crocker[:,:,1],(0,1,7,15,25,45))
    fig.colorbar(bcs1,ax=ax2)
    ax2.set_title("Betti 1")
    plt.yscale('log')
    ax1.set_ylabel(r'Proximity $(\varepsilon)$')
    ax1.set_xlabel('Time (frame)')
    ax2.set_xlabel('Time (frame)')
    plt.tight_layout()
    if save_path == None:
        return fig
    else:
        plt.savefig(save_path)
        plt.close()
        
def plot_crocker_highres(crocker,prox_vec,
                 save_path=None):
    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(12,8),dpi=400)
    bcs0 = ax1.contourf(*np.meshgrid(range(1,crocker.shape[0]+1),
                                     prox_vec,indexing='ij'),
                        crocker[:,:,0],(1,50,100,150,200,250))
    fig.colorbar(bcs0,ax=ax1)
    ax1.set_title("Betti 0",fontsize=30)
    bcs1 = ax2.contourf(*np.meshgrid(range(1,crocker.shape[0]+1),
                                     prox_vec,indexing='ij'),
                        crocker[:,:,1],(0,1,7,15,25,45))
    fig.colorbar(bcs1,ax=ax2)
    ax2.set_title("Betti 1",fontsize=30)
    plt.yscale('log')
    ax1.set_ylabel(r'Proximity $(\varepsilon)$',fontsize=30)
    ax1.set_xlabel('Time (frame)',fontsize=30)
    ax2.set_xlabel('Time (frame)',fontsize=30)
    plt.tight_layout()
    if save_path == None:
        return fig
    else:
        plt.savefig(save_path)
        plt.close()

def plot_crocker_highres_split(crocker,prox_vec,crocker1,prox_vec1,
                 save_path=None):
    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(12,8),dpi=400)
    bcs0 = ax1.contourf(*np.meshgrid(range(1,crocker.shape[0]+1),
                                     prox_vec,indexing='ij'),
                        crocker[:,:,0],(1,50,100,150,200,250))
    fig.colorbar(bcs0,ax=ax1)
    ax1.set_title("Betti 0",fontsize=30)
    plt.yscale('log')
    bcs1 = ax2.contourf(*np.meshgrid(range(1,crocker1.shape[0]+1),
                                     prox_vec1,indexing='ij'),
                        crocker1[:,:,1],(0,1,7,15,25,45))
    fig.colorbar(bcs1,ax=ax2)
    ax2.set_title("Betti 1",fontsize=30)
    plt.yscale('log')
    ax1.set_ylabel(r'Proximity $(\varepsilon)$',fontsize=30)
    ax1.set_xlabel('Time (frame)',fontsize=30)
    ax2.set_xlabel('Time (frame)',fontsize=30)
    plt.tight_layout()
    if save_path == None:
        return fig
    else:
        plt.savefig(save_path)
        plt.close()
        
def custom_metric(row1,row2):

    dist = row1[2]-row2[2]
    if dist<-180:
        dist += 360
    if dist>180:
        dist-=360
    dist = np.abs(dist)
    total_dist = np.sqrt((row1[0]-row2[0])**2 + (row1[1]-row2[1])**2) + (dist/180)*0.1*5
    return total_dist


