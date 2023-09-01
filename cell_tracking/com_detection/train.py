from unet import UNet
from data import CellData
import tensorflow as tf
import glob
import os
import numpy as np
import time
import pickle

#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#set_session(sess)

#Initialize network
net = UNet(log_dir='/home/mars/external4/scratch_oneshot/')
net.build_model()

#Location of directory containing image data
image_dir = '/home/mars/Projects/scratch_assays_scott/'

#Get image paths for training based on COMs filenames
train_coms = ['./labeled_scratch_data/scratch3t3_1nMPDGF1_w2Cy5_s36_t1.json']
train_ims = [os.path.join(image_dir,
                          os.path.splitext(os.path.basename(p))[0]+'.TIF')
             for p in train_coms]

#initialize data handler and get generator
train_dh = CellData(train_ims,max_coms=100,coms_paths=train_coms,
                    sample_dim=128,clip=-0.1,target_type='p_map',
                    random_train=2000) #Using 2000 random tiles and -10% clip
train_steps = train_dh.get_iters()
train_gen = train_dh.train_generator(split=None)

#Train for 5000 epochs. Other training settings are in the UNet.train method
beg = time.time()
hist = net.train(train_gen,train_steps,None,None,epochs=5000)
end = time.time()
print("Time to train = "+str(end-beg)+" seconds")
