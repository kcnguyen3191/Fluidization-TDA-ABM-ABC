import tensorflow as tf
import pdb
import numpy as np

def chamfer_dist_pad(out_shape):
    '''
    A chamfer distance loss function for neural network training.

    This version takes the padded tensors and only masks the distances FROM
    the padded trues to the nearest pred. Thus it must be used with a bounded
    output activation in order to avoid predictions "creeping" toward (-1,-1)

    out_shape (tuple): the shape of the network output
    '''
    def BruteForceChamfer(true,pred):
        num_pred = out_shape[0]
        num_true = out_shape[0]
        d = out_shape[1]
        true, mask = tf.split(true,[d,1],axis=2)
        mask = tf.squeeze(mask,axis=2)
        true_tile = tf.tile(tf.expand_dims(true,2), (1,1,num_pred,1)) 
        pred_tile = tf.tile(tf.expand_dims(pred,1), (1,num_true,1,1))
        pairwise = tf.norm(tf.subtract(true_tile,pred_tile), axis=3)
        dist1 = tf.reduce_min(pairwise,axis=1)
        dist1 = tf.reduce_mean(dist1,axis=1)
        dist2 = tf.reduce_min(pairwise,axis=2)
        dist2 = tf.multiply(dist2,mask)
        dist2 = tf.divide(tf.reduce_sum(dist2,axis=1),
                          tf.reduce_sum(mask,axis=1))
        cham_dist = tf.add(dist1,dist2)

        return cham_dist
    return BruteForceChamfer

def chamfer_dist_ragged(out_shape):
    '''
    A chamfer distance loss function for neural network training.

    This version uses tf.ragged tensors and operations to only compute the
    actual distances. This should work properly on any (unbounded) outputs,
    but requires updated tensorflow (version >=1.14).

    out_shape (tuple): the shape of the network output
    '''
    def BruteForceChamfer(true,pred):
        num_pred = out_shape[0]
        d = out_shape[1]
        true, mask = tf.split(true,[d,1],axis=2)
        mask = tf.squeeze(mask,axis=2)
        mask = tf.cast(mask,dtype=tf.bool)
        m_true = tf.ragged.boolean_mask(true,mask)
        true_tile = tf.tile(tf.expand_dims(m_true,2),(1,1,num_pred,1)) 
        pred_tile = tf.tile(tf.expand_dims(pred,1), (1,num_pred,1,1))
        pred_tile = tf.ragged.boolean_mask(pred_tile,mask)
        pairwise = tf.sqrt(tf.reduce_sum(
            tf.square(tf.subtract(true_tile,pred_tile)),
                           axis=3))
        dist1 = tf.reduce_mean(tf.reduce_min(pairwise,axis=1),axis=1)
        dist2 = tf.reduce_mean(tf.reduce_min(pairwise,axis=2),axis=1)
        cham_dist = tf.add(dist1,dist2)

        return cham_dist
    return BruteForceChamfer
