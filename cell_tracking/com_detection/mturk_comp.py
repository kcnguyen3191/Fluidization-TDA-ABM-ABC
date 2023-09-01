import pandas as pd
import numpy as np
from evaluation import chamfer_distance
from PIL import Image
import os
import matplotlib.pyplot as plt

#Path to ground-truth Mechanical Turk annotations
GT_PATH = '../labeled_scratch_data/s27_t60_quarter1_MTurk.csv'
#Paths to location DataFrame pickles (see locate.py)
DF_PATHS = ['./trackpy_locations/stage27_frame60_raw_locations_quarter1.pkl',
            './trackpy_locations/stage27_frame60_sig_locations_quarter1.pkl',
            './trackpy_locations/stage27_frame60_pmap_locations_quarter1.pkl']
#Path to (contrast-corrected) image quarter (see image_correction.py)
IMAGE_PATH = './processed_TIFs/scratch3t3_1nMPDGF1_w2Cy5_s27_t60_corrected_quarter1.png'
im = Image.open(IMAGE_PATH)
im = np.array(im)

gt_df = pd.read_csv(GT_PATH,header=0)
gt_dict = {str(k):eval(v) for (k,v) in
           zip(gt_df["WorkerId"],gt_df["Answer.annotatedResult.keypoints"])}
for p in DF_PATHS:
    pred_df = pd.read_pickle(p)
    pred = np.column_stack((pred_df.x.to_numpy(),pred_df.y.to_numpy()))
    for (gt_id,keypoints) in gt_dict.items():
        plt.figure(figsize=(6.72,5.12))
        plt.imshow(im,cmap='gray',origin='upper',interpolation='none')
        true = np.array([(subdict['x'],subdict['y']) for subdict in keypoints])
        chamf = chamfer_distance(true,pred)
        print("Error from {0} to {1} = {2:.3f}".format(
                os.path.basename(p),gt_id,chamf))
        plt.plot(true[:,0],true[:,1],'.',ms=6,c='tab:blue')
        plt.plot(pred[:,0],pred[:,1],'.',ms=2,c='tab:orange')
        plt.tick_params(bottom=False,left=False,
                        labelbottom=False,labelleft=False)
        plt.legend(["True","Pred"],loc='upper left')
        plt.tight_layout()
        plt.show()
