import numpy as np
from itertools import permutations
from math import factorial
import scipy.optimize as opt
from functools import partial
import time
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os
import glob
import imageio as io
from itertools import repeat
from ripser import ripser
import scipy
import matplotlib as mpl
from functools import partial
import concurrent.futures
from Scripts.DorsognaNondim import *

def compute_betti(barcodes,prox_vec):
    '''Computes the Betti curve of barcodes given a list of proximity values.
    '''
    betti_stack = [
        [np.sum(np.logical_and(b[:,0]<prox,b[:,1]>=prox)) for prox in prox_vec]
                   for b in barcodes]
    return np.stack(betti_stack,axis=1)

def compute_betti_list(j,traj_df,prox_vec,data_cols,betti):
    
    #Get relevant data from df
    data = traj_df[traj_df['frame']==j]
    data = data[list(data_cols)].to_numpy()
    #print("\r{0} points in frame {1}".format(data.shape[0],j),end='')
    if data.shape[0] == 0:
        print("\r{0} points in frame {1}".format(data.shape[0],j),end='')
    #Compute barcodes
    barcodes = ripser(data,maxdim=max(betti))['dgms']
    #Replace inf with maximum H1 death value
    barcodes[0][barcodes[0] == np.inf] = max(barcodes[1].max(),prox_vec.max())
    
    return compute_betti(barcodes,prox_vec)[:,betti]

def compute_crocker(traj_df,frame_vec,prox_vec,num_run=None,
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
    #Make crocker metric from compute_crocker function
    betti_list_func = partial(compute_betti_list,
                              traj_df=traj_df,
                              prox_vec=prox_vec,
                              data_cols=data_cols,
                              betti=betti)
    betti_curves = []
    if num_run > 1:
        for j in frame_vec:#for each desired frame
            betti_curves.append(betti_list_func(j))
    else: # Parallel computing for betti curves
        # Parallel computing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            betti_curves = executor.map(betti_list_func, frame_vec)
    
    crocker = np.stack(betti_curves,axis=0)
    
    return crocker

def density_metric(df,frame_vec,num_rect=100):
    #Define needed components
    cutoffs = np.linspace(0.,100.,num_rect+1)
    area = (cutoffs[1]-cutoffs[0])*100

    #Compute density estimate for each desired frame
    densities = []
    for f in frame_vec:
        frame = df.loc[df['frame'] == f]
        cell_counts = [sum((frame['y'] > i) & (frame['y'] <= j))
                       for (i,j) in zip(cutoffs,cutoffs[1:])]
        densities.append([c/area for c in cell_counts])

    return np.array(densities)


def obj_func(free_params,fixed_param_dict,ic_vec,time_vec,metric,true_metric,tracker_file=None,
             verbose=False):
    '''Objective function wrapper to be used for parameter estimation.

    Inputs:
        free_params (1d ndarray): SORTED parameter guesses to be optimized
        fixed_param_dict (dict): fixed parameters (if any) to be used in model

        ic_vec (1d ndarray): Concatenated vector of initial conditions
        time_vec (1d ndarray): Vector of times for simulation purposes

        metric (callable): metric function that describes global behavior
        true_metric (ndarray): Ground-truth result from "metric"

    Kwarg: verbose (bool): Whether to print status while running

    Output: loss (float) loss between metric arrays
    '''
    #Return inf if any free parameter is negative
    if any([p < 0 for p in free_params]):
        return np.inf
    #Create dictionary of all parameters from union of free and fixed
    #p_names = {'sigma','alpha','beta','cA','cR','lA','lR'}
    p_names = {'sigma','alpha','beta','c','l'}
    p_names.difference_update(fixed_param_dict.keys())
    p_names = sorted(list(p_names))
    param_dict = fixed_param_dict.copy()
    param_dict.update(dict(zip(p_names,free_params)))
#    if verbose:
    print("Current parameters:",param_dict)
    
#    if tracker_file is not None:
#        # Append-adds at last
#        with open(tracker_file, "a") as f:
#            f.write("C=" + str(param_dict['c']) + "l=" + str(param_dict['l']) + "\n")
##        file1 = open(tracker_file, "a")  # append mode
##        file1.write("C=" + param_dict['c'] + "l=" + param_dict['l'] + "\n")
##        file1.close()
        
    #Simulate using param_dict and true position and velocity as ICs
#     print(time_vec)
    simulator = DorsognaNondim(**param_dict)
    simulator.ode_rk4(ic_vec,time_vec[0],time_vec[-1],time_vec[1]-time_vec[0])
    simu = simulator.results_to_df(time_vec)

    try:
        #Get pred metric from simulated data
        pred_metric = metric(simu)

        if len(true_metric.shape) > 2:
            max0 = np.log10(np.max(true_metric[:,:,0]))
            true0 = np.log10(true_metric[:,:,0],where=true_metric[:,:,0]>0)
            pred0 = np.log10(pred_metric[:,:,0],where=pred_metric[:,:,0]>0)
            
            max1 = np.log10(np.max(true_metric[:,:,1]))
            true1 = np.log10(true_metric[:,:,1],where=true_metric[:,:,1]>0)
            pred1 = np.log10(pred_metric[:,:,1],where=pred_metric[:,:,1]>0)
            
            loss = np.sum(((true0 - pred0)/max0)**2) + np.sum(((true1 - pred1)/max1)**2)
#            loss = np.sum((np.log10(true_metric[:,:,0],where=true_metric[:,:,0]>0)-np.log10(pred_metric[:,:,0],where=pred_metric[:,:,0]>0))**2)/np.log10(np.max(true_metric[:,:,0]))**2 + np.sum((np.log10(true_metric[:,:,1])-np.log10(pred_metric[:,:,1]))**2)/np.log10(np.max(true_metric[:,:,1]))**2
#            loss = np.sum((np.log10(true_metric[:,:,0])-np.log10(pred_metric[:,:,0]))**2)/np.log10(200)**2 + np.sum((np.log10(true_metric[:,:,1])-np.log10(pred_metric[:,:,1]))**2)/np.log10(45)**2
        else:
            loss = np.sum((np.log10(true_metric)-np.log10(pred_metric))**2/np.max(np.log10(true_metric))**2)
        #Return GMSE between metrics
#         residuals = true_metric-pred_metric
#         loss = np.sum(np.divide(residuals,np.abs(pred_metric)**GAMMA,
#                         out=residuals.astype(np.float64),where=pred_metric!=0)**2)
    except:
        loss = 1e64
    if verbose:
        print("Current loss: {:.2e}".format(loss))
    return loss

def opt_par_search(true_df,estim_params,fixed_params,frame_vec,metric,tracker_file=None,
                   output=sys.stdout):
    '''Optimize parameters using Nelder-Mead and ground-truth data'''
    #Get needed information from true_df
    time_vec = np.sort(true_df.t.unique())
    true_metric = metric(true_df,frame_vec)

    ic_df = true_df.loc[true_df['frame']==0]
    ic_vec = (ic_df.x.to_numpy(dtype=np.float64),
              ic_df.y.to_numpy(dtype=np.float64),
              ic_df.vx.to_numpy(dtype=np.float64),
              ic_df.vy.to_numpy(dtype=np.float64))
    ic_vec = np.hstack(ic_vec)
    verbose = True
    #Redefine metric with frame_vec
    metric = partial(metric,frame_vec=frame_vec)

    #Make vector of free parameters
    #x_init = np.stack([estim_params[k] for k in sorted(estim_params.keys())],
    #                  axis=1)
    #For each possible initial simplex...
    optim_dicts = []
    for p_idx,p in enumerate(permutations(range(len(estim_params)))):
        output.write("Initial simplex {0} of {1}:\n".format(p_idx+1,
                                                  factorial(len(estim_params))))
        #Get simplex
        vertex = list(0 for _ in range(len(estim_params)))
        simplex = [[interval[vertex[i]] for
                    i,interval in enumerate(estim_params.values())]]
        for i in p:
            vertex[i] += 1
            simplex.append([interval[vertex[i]] for
                            i,interval in enumerate(estim_params.values())])
        output.write(str(simplex)+'\n')
        simplex = np.array(simplex)

        #Get optimal free parameter values
        beg = time.time()
        res = opt.minimize(obj_func,[None]*len(estim_params),#Hack for x0 arg
                        args=(fixed_params,ic_vec,time_vec,metric,true_metric,verbose),
                        method='Nelder-Mead',
                        options={'initial_simplex':simplex,
                                 'maxiter':100*len(estim_params)})
        end = time.time()
        #Print results
        output.write(res.message+'\n')
        output.write("Total optimization time = {0:.2f} minutes\n".format(
                (end-beg)/60))
        output.write("{0} optimization steps; {1} metric evaluations\n".format(
                res.nit,res.nfev+1))
        output.write("Minimum loss = {0}\n".format(res.fun))
        output.write("Optimal parameter values:\n")
        optim_par_dict = dict(zip(sorted(estim_params.keys()),res.x))
        optim_par_dict['loss'] = res.fun
        optim_par_dict['time'] = (end-beg)/60
        optim_dicts.append(optim_par_dict)
        for par,value in optim_par_dict.items():
            output.write("{0} = {1:.7f}\n".format(par,value))
        
    return optim_dicts

def opt_par_search_ran(true_df,estim_params,fixed_params,frame_vec,metric,init_pars,
                       output=sys.stdout):
    '''Optimize parameters using Nelder-Mead and ground-truth data'''
    #Get needed information from true_df
    time_vec = np.sort(true_df.t.unique())
    true_metric = metric(true_df,frame_vec)
    ic_df = true_df.loc[true_df['frame']==0]
    ic_vec = (ic_df.x.to_numpy(dtype=np.float64),
              ic_df.y.to_numpy(dtype=np.float64),
              ic_df.vx.to_numpy(dtype=np.float64),
              ic_df.vy.to_numpy(dtype=np.float64))
    ic_vec = np.hstack(ic_vec)
    verbose = True
    #Redefine metric with frame_vec
    metric = partial(metric,frame_vec=frame_vec)
    
    bnds = np.zeros((2,len(estim_params)))
    for idx, key in enumerate(estim_params.keys()):
        bnds[:,idx] = estim_params[key]
    bnds = scipy.optimize.Bounds(bnds[0,:],bnds[1,:])
    
    optim_dicts = []
#     for r_idx in range(num_run):
#         output.write("Initial parameter {0} of {1}:\n".format(r_idx+1,
#                                                   num_run))

    #Get optimal free parameter values
    beg = time.time()
    res = opt.minimize(obj_func,init_pars,#Hack for x0 arg
                    args=(fixed_params,ic_vec,time_vec,metric,true_metric,verbose),
                    method='Nelder-Mead',
                    options={'xatol':1e-4,
                             'fatol':1e-4,
                             'maxiter':100*len(estim_params)})
    end = time.time()
#         #Print results
    output.write(res.message+'\n')
    output.write("Total optimization time = {0:.2f} minutes\n".format(
            (end-beg)/60))
    output.write("{0} optimization steps; {1} metric evaluations\n".format(
            res.nit,res.nfev+1))
    output.write("Minimum loss = {0}\n".format(res.fun))
    output.write("Optimal parameter values:\n")
    optim_par_dict = dict(zip(sorted(estim_params.keys()),res.x))
    optim_dicts.append(optim_par_dict)
    for par,value in optim_par_dict.items():
        output.write("{0} = {1:.7f}\n".format(par,value))

    return optim_dicts
