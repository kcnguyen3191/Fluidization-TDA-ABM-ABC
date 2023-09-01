import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy import io
from sklearn.manifold import TSNE
matplotlib.rcParams.update({'font.size': 5})

Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)

list_tuples = []
#for C_idx, C in enumerate(Cs):
#    for L_idx, L, in enumerate(Ls):
#        list_tuples.append((C_idx, L_idx))

C_idx = 0
L_idx = 1
count = 1
for C_idx, C in enumerate(Cs):
    for L_idx, L in enumerate(Ls):
        CROCKER_PATH = './Simulated_Grid/ODE/Cidx_'+str(C_idx+1).zfill(2)+'_Lidx_'+str(L_idx+1).zfill(2)+'/run_1/crocker_angles.npy'
        crocker = np.load(CROCKER_PATH, allow_pickle=True)
        crocker = np.asarray(crocker, dtype='float64')

        # T-SNE on the last betti numbers in time
        b0_end  = crocker[:,:,0]
        b1_end  = crocker[:,:,1]
        b01_end = crocker[:,:,:]
        
        if count == 1:
            b0_end_mat  = b0_end.reshape((-1,1))
            b1_end_mat  = b1_end.reshape((-1,1))
            b01_end_mat = b01_end.reshape((-1,1))
        else:
            b0_end_mat  = np.concatenate((b0_end_mat,b0_end.reshape((-1,1))),axis=1)
            b1_end_mat  = np.concatenate((b1_end_mat,b1_end.reshape((-1,1))),axis=1)
            b01_end_mat = np.concatenate((b01_end_mat,b01_end.reshape((-1,1))),axis=1)
            
        count += 1

b0_end_mat  = b0_end_mat.T
b0_end_mat  = b0_end_mat/b0_end_mat.max()
b1_end_mat  = b1_end_mat.T
b1_end_mat  = b1_end_mat/b1_end_mat.max()
b01_end_mat = np.concatenate((b0_end_mat,b1_end_mat),axis=1)

num_run = 1
for irun in range(num_run):
    print(irun)
    b0_end_embedded  = TSNE(n_components=3,verbose=2).fit_transform(b0_end_mat)
    b1_end_embedded  = TSNE(n_components=3,verbose=2).fit_transform(b1_end_mat)
    b01_end_embedded = TSNE(n_components=3,verbose=2).fit_transform(b01_end_mat)
    
    results = {}
    # RGB
    results['RGB'] = {}
    results['RGB']['b0']  = b0_end_embedded
    results['RGB']['b1']  = b1_end_embedded
    results['RGB']['b01'] = b01_end_embedded

    np.save('./Simulated_Grid/ODE/tsne_results_full'+str(irun+1)+'_angles.npy',results)
print("Job done!")


tsne_results = np.load('./Simulated_Grid/ODE/tsne_results_full'+str(0+1)+'_angles.npy',allow_pickle=True).item()

b0_tsne  = tsne_results['RGB']['b0']
b1_tsne  = tsne_results['RGB']['b1']
b01_tsne = tsne_results['RGB']['b01']

b0_tsne = b0_tsne.reshape((30,30,-1))
b1_tsne = b1_tsne.reshape((30,30,-1))
b01_tsne = b01_tsne.reshape((30,30,-1))

b0_percentile  = np.percentile(b0_tsne, [1,99], axis=[0,1]) #np.min(b0_tsne[:,:,idx])
b1_percentile  = np.percentile(b1_tsne, [1,99], axis=[0,1]) #np.min(b0_tsne[:,:,idx])
b01_percentile = np.percentile(b01_tsne, [1,99], axis=[0,1]) #np.min(b0_tsne[:,:,idx])

for idx in range(b0_tsne.shape[2]):
    
    b0_tsne[:,:,idx]  = (b0_tsne[:,:,idx] - b0_percentile[0,idx])/(b0_percentile[1,idx] - b0_percentile[0,idx])
    b1_tsne[:,:,idx]  = (b1_tsne[:,:,idx] - b1_percentile[0,idx])/(b1_percentile[1,idx] - b1_percentile[0,idx])
    b01_tsne[:,:,idx] = (b01_tsne[:,:,idx] - b01_percentile[0,idx])/(b01_percentile[1,idx] - b01_percentile[0,idx])

b0_tsne  = np.clip(b0_tsne, 0, 1)
b1_tsne  = np.clip(b1_tsne, 0, 1)
b01_tsne = np.clip(b01_tsne, 0, 1)

fig,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(8,8),dpi=400)

# Imshow plots along the rows for y-axis, then plots along columns for x-axis
# So, swap dimension-0 and dimension-1 to show C on x-axis, and L for y-axis.
ax1.imshow(np.swapaxes(b0_tsne, 0, 1))
ax1.invert_yaxis()
ax1.set_xlabel('C')
ax1.set_ylabel('L')
plt.setp(ax1, xticks=[0, 9, 19, 29], xticklabels=[0.1, 1.0, 2.0, 3.0],
        yticks=[0, 9, 19, 29], yticklabels=[0.1, 1.0, 2.0, 3.0])
ax1.set_title("t-SNE on CROCKER: Betti-0")

ax2.imshow(np.swapaxes(b1_tsne, 0, 1))
ax2.invert_yaxis()
ax2.set_xlabel('C')
plt.setp(ax2, xticks=[0, 9, 19, 29], xticklabels=[0.1, 1.0, 2.0, 3.0])
ax2.set_title("t-SNE on CROCKER: Betti-1")

ax3.imshow(np.swapaxes(b01_tsne, 0, 1))
ax3.invert_yaxis()
ax3.set_xlabel('C')
plt.setp(ax3, xticks=[0, 9, 19, 29], xticklabels=[0.1, 1.0, 2.0, 3.0])
ax3.set_title("t-SNE on CROCKER: Betti-0 and Betti-1")

fig.tight_layout()
fig.savefig('./Simulated_Grid/ODE/tsne_crocker_heatmap_angles.pdf',bbox_inches='tight')
plt.show()

b0_tsne_color = np.squeeze(b0_tsne.reshape((-1,1,3)))
b1_tsne_color = np.squeeze(b1_tsne.reshape((-1,1,3)))
b01_tsne_color = np.squeeze(b01_tsne.reshape((-1,1,3)))

mat_tsne = {}
mat_tsne['norm_tsne'] = {}
mat_tsne['norm_tsne']['b0_tsne'] = np.swapaxes(b0_tsne, 0, 1)
mat_tsne['norm_tsne']['b1_tsne'] = np.swapaxes(b1_tsne, 0, 1)
mat_tsne['norm_tsne']['b01_tsne'] = np.swapaxes(b01_tsne, 0, 1)

b0_tsne  = tsne_results['RGB']['b0']
b1_tsne  = tsne_results['RGB']['b1']
b01_tsne = tsne_results['RGB']['b01']

mat_tsne['tsne'] = {}
mat_tsne['tsne']['b0_tsne'] = np.swapaxes(b0_tsne, 0, 1)
mat_tsne['tsne']['b1_tsne'] = np.swapaxes(b1_tsne, 0, 1)
mat_tsne['tsne']['b01_tsne'] = np.swapaxes(b01_tsne, 0, 1)

scipy.io.savemat('./Simulated_Grid/ODE/tsne_mat_angles.mat',mat_tsne)
