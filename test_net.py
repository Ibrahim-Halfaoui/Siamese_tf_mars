""" Test function for two given images for person Re-ID model
@ author: ibrahim Halfaoui
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import model 
from dloader import dataloader
import os
dir_ = os.getcwd()


def test_net(image1, image2):
    # Display images 
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    im1_plt = plt.imshow(image1[0])
    a.set_title('image1')
    b = fig.add_subplot(1,2,2)
    im2_plt = plt.imshow(image2[0])
    b.set_title('image2')   

    # Prepare test 
    siamese = model.siamese()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    # Restore from checkpoint
    restore_path = dir_ + '/model/model'   
#    restore_path = restore_path.split(".")[0]
    saver.restore(sess, restore_path)
    
    # Uncomment to check with a loop test
#    data = dataloader('/mnt/disk4/tensorflow_projects/person_RID/dataset/bbox_test/', 'test', 1)
#    for i in range(50):
#            image1, l1, image2, l2 = data.get_batch()
#            if (l2 == l1):
#                print('test images are similar')
#            else:
#                print('test image are NON similar')
    
    # Unsqueeze images 
#    image1 = np.expand_dims(image1, axis=0)
#    image2 = np.expand_dims(image2, axis=0)
    
    # Run inf
    embed1, embed2 = sess.run([siamese.o1, siamese.o2],
                            feed_dict={
                            siamese.x1: image1,
                            siamese.x2: image2,                                                      
                            })
    # Distance computation
    print('\n')
    distance = np.sqrt(np.sum(np.square(embed1 - embed2)))   
    print('Similarity Metric = ', distance) 
    print('\n')        
    print('Test is done.')
    return 0 
