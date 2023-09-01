from unet import UNet
from data import CellData
from evaluation import chamfer_distance,detect_peaks,make_pmap
import os
import numpy as np
import pickle
import time
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

'''
Script for inferring on raw image data and comparing to labels using methods
from evaluation.py. Puts prediction plots in net.log_dir.
'''

#Initialize network with desired logging directory
net = UNet(log_dir='/home/mars/Carter/eval_scratch_assay/')
net.build_model()

#Location of directory containing image data
IMAGE_DIR = '/home/mars/Project/scratch_assays_scott/'

#Get image paths for inference based on provided COMs files
test_coms = ['../labeled_scratch_data/scratch3t3_1nMPDGF1_w2Cy5_s36_t64.json']
test_ims = [os.path.join(IMAGE_DIR,
                         os.path.splitext(os.path.basename(p))[0]+'.TIF')
            for p in test_coms]

#Initialize data handler with desired settings and make generator
test_dh = CellData(test_ims,max_coms=100,coms_paths=test_coms,
                   sample_dim=128,overlap=0.1)
test,test_steps = test_dh.get_iters()
test_gen = test_dh.infer_generator(test)

#Get image predictions using provided weights
beg = time.time()
pmap_preds = net.infer(test_gen,
    weights='/home/mars/external4/scratch_oneshot/best_train_weights.h5')
end = time.time()
print("Total inference time: {0:.2f} seconds".format(end-beg))

#Loop through predictions and true COMs
for pred_map,image_path,com_path in zip(pmap_preds,test_ims,test_coms):
    coms = test_dh.read_json(com_path)#get true COM locations
    #Compute MSE between p-maps
    true_map = make_pmap(coms,(pred_map.shape[1],pred_map.shape[0]))
    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True)
    ax1.imshow(true_map)
    ax2.imshow(pred_map)
    save_path = os.path.join(net.log_dir,os.path.splitext(
                    os.path.basename(image_path))[0]+"_pmap_interactive.pkl")
    pickle.dump(fig,open(save_path,'wb'))
    plt.close()
    mse = np.mean((pred_map-true_map)**2)
    print("P-map MSE = {0:.5f}".format(mse))
    #Compute chamfer distance between COM locations
    pred_x,pred_y = detect_peaks(pred_map,
                                 clip_level=0.07,threshold=0.001,peak_size=9)
    chamf = chamfer_distance(np.array(coms),np.stack([pred_x,pred_y],axis=-1))
    print("COM location chamfer distance = {0:.2f}".format(chamf))
    #Plot pred and true COMs side-by-side on original image
    im = Image.open(image_path)
    im = np.array(im)
    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(25,11))
    fig.subplots_adjust(left=0.04,right=0.97,wspace=0.05)
    ax1.imshow(im,cmap='gray',vmin=0,vmax=4095)#plot the original image
    ax1.plot(pred_x,pred_y,
             'r.',ms=2,scalex=False,scaley=False)
    ax1.set_title("Predictions")#plot predicted COM peaks on image
    ax2.imshow(im,cmap='gray',vmin=0,vmax=4095)
    ax2.plot([x for (x,y) in coms],[y for (x,y) in coms],
             'r.',ms=2,scalex=False,scaley=False)
    ax2.set_title("Annotations")#plot true COMs on image
    save_path = os.path.join(net.log_dir,os.path.splitext(
                    os.path.basename(image_path))[0]+"_scatter.png")
    plt.savefig(save_path)
    plt.close()
