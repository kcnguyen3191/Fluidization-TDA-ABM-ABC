"""Grid search for finding ideal Trackpy 'diameter' parameter"""
import trackpy as tp
import numpy as np
from PIL import Image

def quarter(image):
    im_array = np.array(image)
    quarters = [im_array[:im_array.shape[0]//2,:im_array.shape[1]//2]]
    quarters.append(im_array[:im_array.shape[0]//2,im_array.shape[1]//2:])
    quarters.append(im_array[im_array.shape[0]//2:,:im_array.shape[1]//2])
    quarters.append(im_array[im_array.shape[0]//2:,im_array.shape[1]//2:])
    return quarters

def diam_grid_search(image,target,
                     d_range=range(3,35,2)):
    best_diff = np.inf
    for d in d_range:
        print("Diameter =",d)
        df = tp.locate(image,d)
        num_cells = df.shape[0]
        print(num_cells,"cells found")
        if abs(target-num_cells) < best_diff:
            best_d = d
            best_diff = abs(target-num_cells)
            keep_df = df.copy()
    return keep_df,best_d

#Which quarter of the image to consider (numbered according to image_correction)
QUARTER = 1
#Which image to use (raw TIF, corrected PNG, p-map PNG, etc)
IMAGE_PATH = './processed_TIFs/scratch3t3_1nMPDGF1_w2Cy5_s41_t60_corrected.png'
#Where to save the Trackpy detection DataFrame pickle
SAVE_PATH = ('./trackpy_locations/'
             'stage41_frame60_sig_locations_quarter{0}.pkl'.format(QUARTER))
#The ground-truth number of cells in the image quarter
GT_NUM_CELLS = 587

im = Image.open(IMAGE_PATH)
a = np.array(im)
qs = quarter(im)
a = qs[QUARTER-1]
df,best_d = diam_grid_search(a,GT_NUM_CELLS)
print("Best diameter =",best_d)
print("Number of cells =",df.shape[0])
df.to_pickle(SAVE_PATH)
