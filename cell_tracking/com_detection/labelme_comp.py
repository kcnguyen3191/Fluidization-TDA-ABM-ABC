import pandas as pd
import json
import numpy as np
from evaluation import chamfer_distance
from PIL import Image
import os
import matplotlib.pyplot as plt

#Path to ground-truth LabelMe annotations
GT_PATH = '../labeled_scratch_data/scratch3t3_1nMPDGF1_w2Cy5_s36_t64.json'
#Paths to location DataFrame pickles (see locate.py)
DF_PATHS = ['./trackpy_locations/stage36_frame64_raw_locations.pkl',
            './trackpy_locations/stage36_frame64_sig_locations.pkl',
            './trackpy_locations/stage36_frame64_pmap_locations.pkl']
#Path to (contrast-corrected) image (see image_correction.py)
IMAGE_PATH = './processed_TIFs/scratch3t3_1nMPDGF1_w2Cy5_s36_t64_corrected.png'
im = Image.open(IMAGE_PATH)
im = np.array(im)

gt_annots = json.load(open(GT_PATH,'r'))
for p in DF_PATHS:
    pred_df = pd.read_pickle(p)
    pred = np.column_stack((pred_df.x.to_numpy(),pred_df.y.to_numpy()))
    plt.imshow(im,cmap='gray',origin='upper',interpolation='none')
    true = np.array([(l['points'][0][0],l['points'][0][1])
                     for l in gt_annots['shapes']])
    chamf = chamfer_distance(true,pred)
    print("Error for {0} = {1:.3f}".format(
                os.path.basename(p),chamf))
    plt.plot(true[:,0],true[:,1],'.',ms=5,c='tab:blue')
    plt.plot(pred[:,0],pred[:,1],'.',ms=3,c='tab:orange')
    plt.show()
