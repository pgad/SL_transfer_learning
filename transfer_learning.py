# pgad

# Libraries used
import math
import os
import numpy as np
import scipy

import Tensorflow as tf

# Constants
# SL data has two classes: '1' for strong lenses and '0' for non-lenses

SL_CLASSES = 2 

# The SL images have been cropped to a size of 64 X 64 X 3

IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGGE_SIZE*IMAGE_SIZE*3

# Batch size (must divide the data perfectly)

BATCH_SIZE = 100
EVAL_BATCH_SIZE = 1

# Number of units in the hidden layers

HIDDEN1_UNITS = 32
HIDDEN2_UNITS = 16
HIDDEN3_UNITS = 16
HIDDEN4_UNITS = 16

# Maximum number of training steps

MAX_STEPS = 2000

# Directory to put the training data

TRAIN_DIR = 't'

# Build inference graph
def simpconvNN(images, hidden1_units, hidden2_units):

    # Hidden 1
    # Go from 64 x 64 x 3 to 32 x 32 x 3
    
    with tf.name_scope('maxpool1'):
        hidden1 = tf.nn.max_pool(images, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
 
    # Hidden 2
    # Go from 32 x 32 x 3 to 16 x 16 x 64
    
    with tf.name_scope('conv2d_1'):
        weights = tf.Variable(
            tf.random_normal([3, 3, 3, 64])
            )
        bias = tf.Variable(
            tf.random_normal([64])
        )
        hidden2 = tf.nn.conv2d(hidden1, weights, strides = [1, 2, 2, 1])
        hidden2 = tf.nn.bias_add(hidden2, bias)
        hidden2 = tf.nn.softplus(hidden2)
        
    # Hidden 3
    # Go from 16 x 16 x 64 to 16 x 16 x 32
    
    with tf.name_scope('conv2d_2'):
        weights = tf.Variable(
            tf.random_normal([3, 3, 64, 32])
            )
        bias = tf.Variable(
            tf.random_normal([32])
        )
        hidden3 = tf.nn.conv2d(hidden2, weights)
        hidden3 = tf.nn.bias_add(hidden3, bias)    
        hidden3 = tf.nn.softplus(hidden3)
        
    # Hidden 4
    # Go from 16 x 16 x 64 to 16 x 16 x 16
    
    with tf.name_scope('conv2d_3'):
        weights = tf.Variable(
            tf.random_normal([3, 3, 32, 16])
            )
        bias = tf.Variable(
            tf.random_normal([16])
        )
        hidden4 = tf.nn.conv2d(hidden3, weights, strides = [1, 2, 2, 1])
        hidden4 = tf.nn.bias_add(hidden4, bias)
        hidden4 = tf.nn.softplus(hidden4)

    # Hidden 5
    # Go from 16 x 16 x 64 to 8 x 8 x 16
    
    with tf.name_scope('maxpool2'):
        hidden5 = tf.nn.max_pool(hidden4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    # Hidden 6
    # Go from 8 x 8 x 16 to 1024 x 1
    
    with tf.name_scope('flatten'):
        hidden6 = tf.contrib.layers.flatten(hidden5)
        
    # Hidden 7
    
    with tf.name_scope('dense1'):
        hidden7 = tf.layers.dense(hidden6, 128)
        hidden7 = tf.nn.softplus(hidden7)
    
    # Hidden 8

    with tf.name_scope('dense2'):
        hidden8 = tf.layers.dense(hidden7, 32)
        hidden8 = tf.nn.softplus(hidden8)
    
    # Hidden 9
    
    with tf.name_scope('dense3'):
        hidden9 = tf.layers.dense(hidden8,1)
        hidden9 = tf.sigmoid(hidden9)
        
return hidden9        