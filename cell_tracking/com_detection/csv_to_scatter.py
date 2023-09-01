import csv
import glob
import os
import numpy as np
from PIL import Image

'''
Creates scatter plots suitable for input into Trackpy from CSV files.
'''
WIDTH = 1344
HEIGHT = 1024

csv_paths = glob.glob("/home/mars/external4/scratch_oneshot_0817/20_overlap/pred_coms/scratch3t3_1nMPDGF1_w2Cy5_s*.csv")
print(len(csv_paths),"CSVs found.")

save_dir = "/home/mars/Projects/scratch_assays_scott/pred_plots/"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

for path in csv_paths:
    #initialize empty np array of desired size
    array = np.zeros((HEIGHT,WIDTH),dtype=np.uint8)
    #scatter plot each point in the csv
    with open(path,'r',newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            r = int(row[0])
            c = int(row[1])
            array[r,c] = 255
    #create image
    im = Image.fromarray(array,mode='L')
    save_path = os.path.join(save_dir,os.path.splitext(os.path.basename(path))[0]+".png")
    print(save_path)
    im.save(save_path)
