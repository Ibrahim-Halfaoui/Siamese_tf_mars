""" Siamese Archtitecture implementation for person Re-Id using Tensorflow 
@ author: ibrahim Halfaoui
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf

class siamese:
    # Create model
    def __init__(self):
        with tf.variable_scope("siamese") as scope:
        # Init
            self.x1 = tf.placeholder(tf.float32, [None, 256, 128, 3])
            self.x2 = tf.placeholder(tf.float32, [None, 256, 128, 3])
        
        # Set up siamese out of two embedding barnches 
#        with tf.variable_scope("siamese") as scope:
            self.o1 = self.emb_model(self.x1, False)
            scope.reuse_variables()
            self.o2 = self.emb_model(self.x2, True)
           
        # Create loss
            self.y_ = tf.placeholder(tf.float32, [None, 1])
            self.loss = self.contrastive_loss(self.o1, self.o2 , self.y_ , 0.5)

      
    def contrastive_loss(self, model1, model2, y, margin):
        # define loss function
        with tf.name_scope("contrastive-loss"):
            distance = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2) + 1e-10, 1, keepdims=True))
            similarity = y * tf.square(distance)                                           # keep the similar label (1) close to each other  
            dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))   
            return tf.reduce_mean(dissimilarity + similarity) / 2

    
    def emb_model(self, input, reuse=False):
        # Architecture of the branches used to form the siamese
        with tf.name_scope("model"):
            with tf.variable_scope("conv1") as scope:
                net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv2") as scope:
                net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv3") as scope:
                net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv4") as scope:
                net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope("conv5") as scope:
                net = tf.contrib.layers.conv2d(net, 32, [1, 1], activation_fn=None, padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 1024, activation=tf.sigmoid, reuse=reuse)
        return net
     
 
   