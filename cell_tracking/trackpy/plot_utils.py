import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trackpy.filtering import filter_stubs

def plot_histogram(data_frames,save_path,
                   metrics=('length')):
    '''Plots a summary histogram for the provided trajectory dataframes.
    Inputs:
        data_frames (list) - paths to pandas dataframe pickles containing
            trajectory information.
        save_path (str) - where to save the matplotlib figure png

    Kwargs:
        metrics (tuple) - which histograms to make
            length: length (in number of frames) of each trajectory
            distance: distance traveled (in pixels) for each trajectory

    Output: saves .npz of histogram dict to pwd
    '''
    #Initialize histogram dict based on desired metrics to report
    histograms = dict()
    if 'length' in metrics:
        histograms['length'] = np.zeros(128,)
    if 'distance' in metrics:
        histograms['distance'] = np.zeros(100,)

    #Loop through dataframes, creating the desired histograms
    for f in data_frames:
        print(f)
        traj_df = pd.read_pickle(f)
        #traj_df = filter_stubs(traj_df,17)#filter short traj
        if 'length' in metrics:
            traj_lens = [len(traj_df.loc[traj_df['particle']==i]['frame'])
                         for i in traj_df.particle.unique()]
            histograms['length'] += np.histogram(traj_lens,
                                                 bins=128,range=(1,129))[0]
        if 'distance' in metrics:
            traj_dists = [np.sum(np.sqrt(np.sum(
                (traj_df.loc[traj_df['particle']==i][['x','y']].to_numpy()[1:]-
                 traj_df.loc[traj_df['particle']==i][['x','y']].to_numpy()[:-1])**2,
                axis=1))) for i in traj_df.particle.unique()]
            histograms['distance'] += np.histogram(traj_dists,
                                                   bins=100,range=(1,1001))[0]

    #Save the histogram objects for inspection
    np.savez('./traj_histogram.npz',**histograms)

    #Change plot_dict to alter appearance of matplotlib plots
    plot_dict = {'length':{
                    'bar':{'x':range(1,128+1),
                           'height':histograms['length'],
                           'width':0.8},
                    'ylim':(0,15000)},
                 'distance':{
                    'bar':{'x':range(1,1000+1,10),
                           'height':histograms['distance'],
                           'width':8},
                    'ylim':(0,20000)}}
    #Create a subplot for each histogram
    fig,axs = plt.subplots(len(metrics),1,figsize=(13,5*len(metrics)))
    for ax,met in zip(axs,metrics):
        ax.bar(**plot_dict[met]['bar'])
        ax.set_ylim(plot_dict[met]['ylim'])

    plt.savefig(save_path,bbox_inches='tight',pad_inches=0.1)

if __name__ == "__main__":
    #Get list of dataframe pickles using glob
    PICKLE_LIST = glob.glob('./trajectories/*traj_data_filled_in.p')
    plot_histogram(PICKLE_LIST,'./all_traj_histograms.png',
                   metrics=('length','distance'))
