from unet import UNet
from data import CellData
from evaluation import detect_peaks
import time
import os
import glob
import numpy as np
import csv
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_coms_on_tiff(path,pred_map,plot_dir,
                      samp_points=([],[])):
    im = Image.open(path)
    array = np.array(im)
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0, 
                        hspace=0,wspace=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(array,cmap='gray',vmin=0,vmax=4095) #show input image
    plt.scatter(samp_points[0],samp_points[1],c='b',s=5,alpha=0.4)
    pred_x,pred_y = detect_peaks(pred_map,
                                 clip_level=0.07,threshold=0.001,peak_size=9)
    plt.scatter(pred_x,pred_y,c='r',s=1)
    save_path = os.path.join(plot_dir,
                             os.path.splitext(os.path.basename(path))[0]+".png")
    plt.savefig(save_path)
    plt.close()

def save_to_csv(pred_map,path,com_dir):
    com_path = os.path.join(com_dir,
                            os.path.splitext(os.path.basename(path))[0]+".csv")
    pred_x,pred_y = detect_peaks(pred_map,
                                 clip_level=0.07,threshold=0.001,peak_size=9)
    with open(pred_path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(pred_y,pred_x))

'''
Script for running inference on images using a trained network.
Saves predicted p_maps to LOG_DIR
'''

#Set vars and initialize network
LOG_DIR = '/home/mars/external4/scratch_assay_final_pmaps/20_overlap/'
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
IMAGE_DIR = '/home/mars/Projects/scratch_assays_scott/'
net = UNet(log_dir=LOG_DIR)
net.build_model()

#Get inference image data generator
infer_path_list = glob.glob('/home/mars/Projects/scratch_assays_scott/scratch3t3_1nMPDGF1_w2Cy5_s*_t*.TIF')
infer_path_list.sort()
infer_dh = CellData(infer_path_list,max_coms=100,
                    shuffle=False,sample_dim=128,overlap=0.2)
infer,infer_steps = infer_dh.get_iters()
infer_gen = infer_dh.infer_generator(infer)

#Run image inference
beg = time.time()
pmap_preds = net.infer(infer_gen,
    weights='/home/mars/external4/scratch_oneshot/best_train_weights.h5')
end = time.time()
print("Total inference time: {0:.2f} seconds".format(end-beg))

assert len(infer_path_list) == len(pmap_preds),"Unmatched images and preds."
#Write predictions to files in LOG_DIR subdirectory
#pmap_dir = os.path.join(LOG_DIR,'pred_pmaps')
#if not os.path.isdir(pmap_dir):
#    os.mkdir(pmap_dir)
for pred_map,path,samp_points in zip(pmap_preds,infer_path_list,infer):
    #Save pmap as rescaled png
    pred_map = (pred_map*(2**8-1)).astype(np.uint8)
    im = Image.fromarray(pred_map)
    map_path = os.path.join(LOG_DIR,
                            os.path.splitext(os.path.basename(path))[0]+'.png')
    im.save(map_path)
