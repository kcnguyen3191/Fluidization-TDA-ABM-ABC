from PIL import Image
import numpy as np
from skimage.exposure import adjust_sigmoid
import os
import glob

def contrast(image_path):
    im = Image.open(image_path)
    im = np.array(im)
    new_im = adjust_sigmoid(im,cutoff=0.016,gain=400)
    new_im = (new_im/257).astype(np.uint8)
    new_im = Image.fromarray(new_im)
    return new_im

def scale(image_path):
    im = Image.open(image_path)
    im = np.array(im)
    new_im = im.astype(np.float64)/im.max()*255
    new_im = new_im.astype(np.uint8)
    new_im = Image.fromarray(new_im)
    return new_im

def quarter(image):
    im_array = np.array(image)
    quarters = [im_array[:im_array.shape[0]//2,:im_array.shape[1]//2]]
    quarters.append(im_array[:im_array.shape[0]//2,im_array.shape[1]//2:])
    quarters.append(im_array[im_array.shape[0]//2:,:im_array.shape[1]//2])
    quarters.append(im_array[im_array.shape[0]//2:,im_array.shape[1]//2:])
    image_quarters = [Image.fromarray(q) for q in quarters]
    return image_quarters

def crop(image,
         rmin=500,rmax=700,
         cmin=300,cmax=500):
    im_array = np.array(image)
    crop = im_array[rmin:rmax,cmin:cmax]
    print("Dimensions of crop:",crop.shape)
    return Image.fromarray(crop)

#Glob a directory containing time-lapse TIF images
TIF_LIST = glob.glob('/path/to/imagedir/*.TIF')
PROC_DIR = './processed_TIFs/'
#For each base image...
for p in sorted(TIF_LIST):
    ###Apply any number of the above operations
    #Contrast correct
    sig_im = contrast(p) 
    sig_path = os.path.join(PROC_DIR,
        os.path.basename(os.path.splitext(p)[0])+'_corrected.png')
    sig_im.save(sig_path)
    #Scale
    scale_im = scale(p)
    scale_path = os.path.join(PROC_DIR,
        os.path.basename(os.path.splitext(p)[0])+'_scaled.png')
    scale_im.save(scale_path)
    #Crop
    crop_sig = crop(sig_im)
    sig_path = os.path.join(PROC_DIR,
        os.path.basename(os.path.splitext(p)[0])+'_corrected_crop.png')
    sig_im.save(sig_path)
    crop_scale = crop(scale_im)
    scale_path = os.path.join(PROC_DIR,
        os.path.basename(os.path.splitext(p)[0])+'_scaled_crop.png')
    scale_im.save(scale_path)
    #Quarter
    sig_qs = quarter(sig_im)
    scale_qs = quarter(scale_im)
    for i,(sig_q,scale_q) in enumerate(zip(sig_qs,scale_qs)):
        sig_path = os.path.join(PROC_DIR,
            os.path.basename(os.path.splitext(p)[0])+'_corrected_quarter{0}.png'.format(i))
        sig_q.save(sig_path)
        scale_path = os.path.join(TIF_DIR,
            os.path.basename(os.path.splitext(p)[0])+'_scaled_quarter{0}.png'.format(i))
        scale_q.save(scale_path)
