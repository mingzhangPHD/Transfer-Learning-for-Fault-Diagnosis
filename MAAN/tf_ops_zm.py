import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

#from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                        decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
      x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv1d(input_, output_dim, 
           k_l=5, d_l=1, stddev=0.02, seed=1,
           name="conv1d", with_w = False, padding='SAME'):
    
    with tf.variable_scope(name):

        if len(input_.get_shape().as_list()) < 3:
            input_ = tf.reshape(input_, [-1, input_.get_shape().as_list()[1], 1])

        w = tf.get_variable('w', [k_l, input_.get_shape()[-1], output_dim], 
                            initializer = tf.contrib.layers.xavier_initializer(seed = seed))        
        #w = tf.get_variable('w', shape=[k_l, input_.get_shape()[-1], output_dim], 
                            #initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(input_, w, stride=d_l, padding= padding,data_format="NHWC")
        b = tf.get_variable("b", shape=[output_dim], initializer = tf.zeros_initializer())
        #b = tf.get_variable("b", shape=[output_dim], initializer=tf.constant_initializer(0.0))
        #conv = activation_function(tf.nn.bias_add(conv, b))
        conv = tf.nn.bias_add(conv, b)
        
        if with_w:
            return conv, w, b
        else:
            return conv        

    
def max_pool1d(input_, 
               p_s = 2, d_l = 2, 
               name = "pool1d"):
    
    with tf.variable_scope(name):
        pool = tf.layers.max_pooling1d(input_, pool_size=p_s, strides=d_l)
        return pool

def fully_conn(input_,output_dim, 
               stddev=0.02,
               name = "conn1d", with_w = False):
    if input_.shape.ndims > 2:
        input_ = tf.contrib.layers.flatten(input_)
    
    with tf.variable_scope(name):
        
        w = tf.get_variable('w', shape=[input_.get_shape()[-1], output_dim], 
                            initializer = tf.contrib.layers.xavier_initializer(seed = 1))         
        #w = tf.get_variable('w', shape=[input_.get_shape()[-1], output_dim], 
                            #initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", shape=[output_dim], initializer = tf.zeros_initializer())
        #b = tf.get_variable("b", shape=[output_dim], initializer=tf.constant_initializer(0.0))
        #conn = tf.nn.tanh(tf.nn.bias_add(tf.matmul(input_,w),b))
        #conn = tf.nn.relu(tf.nn.bias_add(tf.matmul(input_,w),b))
        conn = tf.nn.bias_add(tf.matmul(input_,w),b)
        
        if with_w:
            return conn, w, b
        else:
            return conn          


def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                              strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable("w", [shape[1], output_size], tf.float32,
                               tf.random_normal_initializer(stddev=stddev))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("b", [output_size],
                           initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
