import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import maximum_filter
from scipy.stats import multivariate_normal

def chamfer_distance(true,pred):
    '''Returns the chamfer distance between the predicted point cloud and
    ground-truth point cloud.
    
    Inputs:
        true: (N x 2 ndarray) ground truth COM locations in (x,y) pairs
        pred: (M x 2 ndarray) predicted COM locations in (x,y) pairs
    Output: (float) chamfer distance (sum of min distances between each point
                        and its nearest neighbor in the other list)
    '''
    dists = cdist(true,pred)
    chamf = np.sum(np.min(dists,axis=0)) + np.sum(np.min(dists,axis=1))
    return chamf

def detect_peaks(p_map,
                 clip_level=0.07,threshold=0,peak_size=5):
    '''Returns the (x,y) locations of the peaks of the sigmoid P-map.
    Input: (ndarray) P-map output from a unet COM detector

    Kwargs:
        clip_level: (float) minimum value to consider for peaks
        threshold: (float) minimum difference between peak and background
        peak_size: (int) size of peak-finding filter

    Output: list of integer pixel locations of peaks
    '''
    p_map = p_map.clip(min=clip_level)
    data_max = maximum_filter(p_map,peak_size)
    maxima = (p_map == data_max)
    data_min = np.full_like(data_max,clip_level)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    y,x = np.where(maxima)
    #x,y = x/p_map.shape[1],y/p_map.shape[0]

    return (x,y)

def make_pmap(coms,pmap_size):
    '''Create P-map from list of COM locations serially. Uses minimal memory.
    Inputs:
        coms: (list) x,y locations of centers of mass
        pmap_size: (tuple) size of P-map to create as (width,height)

    Output: (list) float P-maps to be used as targets
    '''
    p_map = np.zeros((pmap_size[1],pmap_size[0]))
    for c in coms:
        pos = np.stack(np.meshgrid(
                np.arange(1,pmap_size[0]+1),
                np.arange(1,pmap_size[1]+1)),axis=-1)
        p_map = np.maximum(p_map,
            multivariate_normal.pdf(pos,mean=(c[0],c[1]),cov=2))
    p_map /= p_map.max()
    
    return p_map

