import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

##### ARGS #####
#List of frames to use for plotting
TIME_LIST = [20,30,40,50,60,70,80,90,100]
#Image path base string
BASE_PATH = ('./processed_TIFs/'
             'scratch3t3_1nMPDGF1_w2Cy5_s41_t{0}_corrected_crop.png')
#Location of crop within image (see image_correction.py crop for defaults)
X_MIN = 300/1157
X_MAX = 500/1157
Y_MIN = 500/1157
Y_MAX = 700/1157
#How far outside of the crop to look for detections
PAD = 0.005
#Where to save the plots
SAVE_DIR = './detections_on_crops/'
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

#Dataframes containing detections and drifts
df = pd.read_pickle('./final_trajectories/stage_41.pkl')
drift = pd.read_pickle('./final_trajectories/pixel_drift/drift_pixels_41.pkl')

#Loop through frames, loading image and plotting detections
for t in TIME_LIST:
    #Get image
    im_path = BASE_PATH.format(t)
    im = Image.open(im_path)
    im = np.array(im)

    #Get pixel drift for frame
    x_drift = drift.iloc[t-1].x
    y_drift = drift.iloc[t-1].y

    #Get relevant sub-dataframe
    subdf = df.loc[(df['x'] >= X_MIN - x_drift/1157 - PAD) & # > min x val
                   (df['x'] <= X_MAX - x_drift/1157 - PAD) & # < max x val
                   (df['y'] >= Y_MIN - y_drift/1157 - PAD) & # > min y val
                   (df['y'] <= Y_MAX - y_drift/1157 - PAD) & # < max y val
                   (df['frame'] == t-1)] # only current frame

    #Correct for drift correction
    subdf = subdf.assign(x = subdf.x + x_drift/1157,
                         y = subdf.y + y_drift/1157)

    #Plot it
    plt.imshow(im,cmap='gray',extent=[X_MIN,X_MAX,Y_MAX,Y_MIN])
    plt.plot(subdf.x,subdf.y,'.',c='tab:orange',ms=25)
    for x,y,label in subdf[['x','y','particle']].itertuples(index=False):
        plt.annotate(label,(x,y),color='tab:blue',
                     fontweight='bold',ha='center',va='center',size='small')
    save_path = os.path.join(SAVE_DIR,
                             os.path.basename(im_path))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
