import os
import tensorflow as tf
from tensorflow.keras import layers,optimizers,callbacks
from tensorflow.keras.models import Model
from layers import AddCoords
import pickle
import numpy as np
from skimage.transform import resize
import time

class UNet:
    '''
    Unet-type architecture for cell nucleus center detection
    '''

    def __init__(self,input_size=128,
                 log_dir='./'):
        #Size of tile to expect
        self.input_size = input_size
        #Where to save weights, images, metrics, etc
        self.log_dir = log_dir
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        #Where to save the checkpoint weights
        self.checkpoint_path = os.path.join(log_dir,'best_{0}_weights.h5')

    def build_model(self):
        inp = layers.Input(shape=(self.input_size,self.input_size,1))
        inp_coords = AddCoords(self.input_size,self.input_size)(inp)

        x = layers.Conv2D(32,(3,3),padding='same')(inp_coords)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32,(3,3),padding='same')(x)
        down1 = layers.Activation('relu')(x)

        x = layers.Conv2D(64,(3,3),strides=2,padding='same')(down1)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64,(3,3),padding='same')(x)
        down2 = layers.Activation('relu')(x)

        x = layers.Conv2D(128,(3,3),strides=2,padding='same')(down2)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128,(3,3),padding='same')(x)
        down3 = layers.Activation('relu')(x)

        x = layers.Conv2D(256,(3,3),strides=2,padding='same')(down3)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256,(3,3),padding='same')(x)
        down4 = layers.Activation('relu')(x)
        
        x = layers.Conv2D(512,(3,3),strides=2,padding='same')(down4)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(512,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(512,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(256,(2,2),strides=2,padding='same')(x)
        up1 = layers.Activation('relu')(x)

        x = layers.Concatenate()([up1,down4])
        x = layers.Conv2D(256,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(256,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(128,(2,2),strides=2,padding='same')(x)
        up2 = layers.Activation('relu')(x)

        x = layers.Concatenate()([up2,down3])
        x = layers.Conv2D(128,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(64,(2,2),strides=2,padding='same')(x)
        up3 = layers.Activation('relu')(x)

        x = layers.Concatenate()([up3,down2])
        x = layers.Conv2D(64,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(32,(2,2),strides=2,padding='same')(x)
        up4 = layers.Activation('relu')(x)

        x = layers.Concatenate()([up4,down1])
        x = layers.Conv2D(32,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32,(3,3),padding='same')(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(1,(1,1),padding='same')(x)
        out = layers.Activation('sigmoid')(x)

        self.model = Model(inputs=inp,outputs=out)
        self.model.summary()

    def train(self,train_gen,train_steps,val_gen,val_steps,epochs):
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.SGD(0.05,momentum=0.9))
        val_ckpt = callbacks.ModelCheckpoint(self.checkpoint_path.format('val'),
                                             monitor='val_loss',
                                             save_best_only=True,
                                             save_weights_only=True)
        ckpt = callbacks.ModelCheckpoint(self.checkpoint_path.format('train'),
                                             monitor='loss',
                                             save_best_only=True,
                                             save_weights_only=True)
        lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,
                                                 patience=50,min_lr=0.001)
        hist = self.model.fit_generator(train_gen,steps_per_epoch=train_steps,
                                        epochs=epochs,validation_data=val_gen,
                                        validation_steps=val_steps,
                                        callbacks=[ckpt,val_ckpt,lr_reducer])
        pickle.dump(hist.history,open(os.path.join(self.log_dir,'hist.p'),'wb'))
        return hist.history

    #TODO: refactor eval mask capabilities into evaluate.py
    def eval(self,test_gen,
             weights=None,return_preds=False):
        if weights:
            self.model.load_weights(weights)
        else:
            self.model.load_weights(self.checkpoint_path)

        preds = []
        good_preds = []
        for image,coms,mask,mask_idx in test_gen:
            beg = time.time()
            pred = self.model.predict_on_batch(image)
            pred = pred.squeeze()
            if return_preds:
                preds.append(pred)
            pred_x,pred_y = self.detect_peaks(pred,0.07,
                                              threshold=0.001,peak_size=5)
            pred_vals = [mask[int(round(y*self.input_size)),
                              int(round(x*self.input_size))]
                         for (x,y) in zip(pred_x,pred_y)]
            good_preds.extend([v != 0 for v in pred_vals])
            coms = coms.squeeze()
        prec = sum(good_preds)/len(good_preds)
        if return_preds:
            return prec,np.array(preds)
        else:
            return prec

    def infer(self,infer_gen,weights=None):
        if weights:
            self.model.load_weights(weights)
        else:
            self.model.load_weights(self.checkpoint_path)

        pred_maps = []
        for i,(images,locs,tile_size,image_size) in enumerate(infer_gen):
            preds = self.model.predict_on_batch(images)
            preds = preds.squeeze()
            image_map = np.zeros((image_size[1],image_size[0]),
                                dtype=np.float32)
            div = np.zeros_like(image_map)
            for p_map,(col,row) in zip(preds,locs):
                p_map = resize(p_map,(tile_size,tile_size),order=3)
                p = p_map[0:min(tile_size,image_size[1]-row),
                          0:min(tile_size,image_size[0]-col)]
                image_map[row:row+tile_size,col:col+tile_size] += p
                div[row:row+tile_size,col:col+tile_size] += np.ones_like(p)
            image_map = image_map/div
            pred_maps.append(image_map)
        return pred_maps
