import numpy as np
import glob
from PIL import Image
from skimage.measure import label,regionprops
import os
import pickle

'''
Script for getting the stats of a cell dataset using its ground-truth masks.
Writes statistics to a "size_dict" for use in training, testing, etc
'''

dir_string = "/path/to/dataset/directory/*/masks/"
mask_dirs = glob.glob(dir_string)
print("# of dirs found = "+ str(len(mask_dirs)))
cell_dict = dict()

for dir_path in mask_dirs:
    mask_paths = glob.glob(dir_path+'*.png')
    print("# of masks found = " + str(len(mask_paths)))
    major_axes = []
    for im_path in mask_paths:
        im = Image.open(im_path)
        a = np.array(im)
        lbl = label(a)
        obj_list = regionprops(lbl)
        for obj in obj_list:
            if obj.area < 2:
                continue
            major_axes.append(obj.major_axis_length)
    mean = sum(major_axes)/len(major_axes)
    image_path = glob.glob(os.path.join(os.path.dirname(os.path.dirname(dir_path)),
                                        'images','*'))
    assert len(image_path) == 1,"Incorrect number of images: {}".format(
                                    len(image_path))
    image_path = image_path[0]
    cell_dict[image_path] = mean

print("# of means found = " + str(len(cell_dict)))
print("Overall mean = " + str(sum(cell_dict.values())/len(cell_dict)))
print("Max tile size = " + str(int(max(cell_dict.values())*5)))

save_path = './size_dict.p'
pickle.dump(cell_dict,open(save_path,'wb'))
