import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import glob
import imageio as io
import scipy
import matplotlib as mpl

Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)

C_mesh, L_mesh = np.meshgrid(Cs, Ls, indexing='ij')
pars_idc = [(18,4),(7,25),(20,1),(9,6),(5,5),(2,15),(15,7),(25,25),(20,15)]

for pars_idx in pars_idc:
    Cidx, Lidx = pars_idx
    C_true = Cs[Cidx-1]
    L_true = Ls[Lidx-1]
    
    # Get ABC results:
    loss_path = './Simulated_Grid/ODE/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'/run_1/sample_losses_angles.npy'
    sample_losses = np.load(loss_path,allow_pickle=True).item()
    
    
    
    sample_idx = []
    C_vals = []
    L_vals = []
    losses = []
    
    # max_Sample = len(sample_losses)
    samples_idcs = list(sample_losses)
    for iSample in samples_idcs:
        iSample = int(iSample)
        sample_idx.append(iSample)
        losses.append(sample_losses[str(iSample)]['loss'])
        C_vals.append(sample_losses[str(iSample)]['sampled_pars'][3])
        L_vals.append(sample_losses[str(iSample)]['sampled_pars'][4])
    losses = np.array(losses)
    C_vals = np.array(C_vals)
    L_vals = np.array(L_vals)

    nan_idc = np.argwhere(np.isnan(losses))

    losses = np.delete(losses, nan_idc, axis=0)
    C_vals = np.delete(C_vals, nan_idc, axis=0)
    L_vals = np.delete(L_vals, nan_idc, axis=0)
    
    min_loss_idx = np.argmin(losses)
    C_min = C_vals[min_loss_idx]
    L_min = L_vals[min_loss_idx]

    distance_threshold = np.percentile(losses, 1, axis=0)
    
    # Plot NM and ABC results
    belowTH_idc = np.where(losses < distance_threshold)[0]
    C_median = np.median(C_vals[belowTH_idc])
    L_median = np.median(L_vals[belowTH_idc])
    
    min_C = 0.1
    max_C = 3.0
    
    min_L = 0.1
    max_L = 3.0
    
    Cs_plot = np.linspace(min_C,max_C,int(np.ceil((max_C/3.0)*30)))
    Ls_plot = np.linspace(min_L,max_L,int(np.ceil((max_L/3.0)*30)))
    C_mesh_plot, L_mesh_plot = np.meshgrid(Cs_plot, Ls_plot, indexing='ij')
    
    sample_count_map = np.zeros((len(Cs_plot),len(Ls_plot)))
    for belowTH_idx in belowTH_idc:
        Cidx_belowTH = np.argmin(np.abs(Cs_plot-C_vals[belowTH_idx]))
        Lidx_belowTH = np.argmin(np.abs(Ls_plot-L_vals[belowTH_idx]))
        sample_count_map[Cidx_belowTH,Lidx_belowTH] += 1

    fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=400)
    plt.contourf(C_mesh_plot, L_mesh_plot, sample_count_map)
    plt.scatter(C_true,L_true,
                c="w",
                linewidths = 0.5,
                marker = '*',
                edgecolor = "k",
                s = 500,
                label='True')
    plt.scatter(C_median,L_median,
                c="k",
                linewidths = 0.5,
                marker = 'o',
                edgecolor = "k",
                s = 180,
                label='ABC-Median')


    if (Cidx == 20 and Lidx == 1) or (Cidx == 5 and Lidx == 1) or (Cidx == 15 and Lidx == 2):
        ax.legend(fontsize=15)
    ax.set_aspect('equal', adjustable='box')
#     ax.set_title(f' Posterior Density \n True C = {C_true:.02}, True L = {L_true:.02} \n ABC-Threshold = {distance_threshold} ',fontsize=16)
#     plt.title(f'ABC-Posterior Density\n', fontsize=25)
    title_str = 'True C = {0:.02}, True L = {1:.02}'.format(C_true, L_true)
    plt.title(title_str, fontsize=18)#, horizontalalignment='center', x=0.535, y=0.85)
#     plt.suptitle(f'(ABC-Threshold = {distance_threshold})\n', fontsize=15,  y=0.5)
    plt.xlabel("C", fontsize=20)
    plt.ylabel("L", fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    if (Cidx == 25 and Lidx == 15) or (Cidx == 25 and Lidx == 25):
        pass
    else:
        ax.set_xticks([0.1, 1.0, 2.0, 3.0])
        ax.set_yticks([0.1, 1.0, 2.0, 3.0])
    
    fig.tight_layout()
    fig.savefig('./Simulated_Grid/ODE/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'/run_1/ABC_'+'Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_angles.pdf',bbox_inches='tight')
