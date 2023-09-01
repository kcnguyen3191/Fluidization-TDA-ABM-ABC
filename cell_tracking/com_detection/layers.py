import tensorflow as tf
from tensorflow.keras.layers import Layer

# Copyright (c) 2019 Uber Technologies, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class AddCoords(Layer):
    def __init__(self,x_dim,y_dim,**kwargs):
        self.x_dim = x_dim
        self.y_dim = y_dim
        super(AddCoords,self).__init__(**kwargs)

    def build(self,input_shape):
        super(AddCoords,self).build(input_shape)

    def call(self,x):
        batch_size_tensor = tf.shape(x)[0]  # get batch size

        xx_ones = tf.ones([batch_size_tensor, self.x_dim], 
                          dtype=tf.int32)                       # e.g. (batch, 64)
        xx_ones = tf.expand_dims(xx_ones, -1)                   # e.g. (batch, 64, 1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0), 
                            [batch_size_tensor, 1])             # e.g. (batch, 64)
        xx_range = tf.expand_dims(xx_range, 1)                  # e.g. (batch, 1, 64)


        xx_channel = tf.matmul(xx_ones, xx_range)               # e.g. (batch, 64, 64)
        xx_channel = tf.expand_dims(xx_channel, -1)             # e.g. (batch, 64, 64, 1)


        yy_ones = tf.ones([batch_size_tensor, self.y_dim], 
                          dtype=tf.int32)                       # e.g. (batch, 64)
        yy_ones = tf.expand_dims(yy_ones, 1)                    # e.g. (batch, 1, 64)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0),
                            [batch_size_tensor, 1])             # (batch, 64)
        yy_range = tf.expand_dims(yy_range, -1)                 # e.g. (batch, 64, 1)

        yy_channel = tf.matmul(yy_range, yy_ones)               # e.g. (batch, 64, 64)
        yy_channel = tf.expand_dims(yy_channel, -1)             # e.g. (batch, 64, 64, 1)

        xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)

        ret = tf.concat([x,xx_channel,yy_channel],axis=-1)    # e.g. (batch, 64, 64, c+2)

        return ret

    def compute_output_shape(self,input_shape):
        num_channels = input_shape[-1]+2
        return (*input_shape[:-1],num_channels)
