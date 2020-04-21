""" Siamese train function implementation for person Re-Id using Tensorflow 
@ author: ibrahim Halfaoui
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import model 
from dloader import dataloader
import os
dir_ = os.getcwd()

def train_net(dataset_path, mode, batch_size, epochs):
    # Prepare model 
    siamese = model.siamese()
    loss = siamese.loss
   
    # Setup Optimizer
    global_step = tf.Variable(0, trainable=False)
    train_step = tf.train.AdamOptimizer(0.001, 0.90, 0.999, 1e-8, use_locking=False).minimize(loss, global_step=global_step)

    # Prepare saver
    saver = tf.train.Saver()
    
   
    # Prepare session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Uncomment For Resuming training
#        load = False
#        model_ckpt = './'
#        if os.path.isfile(model_ckpt):
#            input_var = None
#            while input_var not in ['yes', 'no']:
#                input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
#            if input_var == 'yes':
#                load = True
#        if load: saver.restore(sess, './model')

        # setup tensorboard    
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        log_dir = dir_ + '/logs2/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir) 
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        save_dir = dir_ + '/model2/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir) 
        
        # Call Dataloader
        data = dataloader(dataset_path, mode, batch_size)
        
        # Visualize infos for training
        steps_per_epoch =  int(data.total_size/batch_size) 
        max_numb_iter = epochs * steps_per_epoch 
        print('\n')
        print('Max numb of iterations:', max_numb_iter)
        print('Start training...\n')

        
        # Start the training loop
        for step in range(max_numb_iter): 
            
             batch_x1, batch_y1, batch_x2, batch_y2 = data.get_batch()            
             batch_y = (batch_y1 == batch_y2)
             batch_y = batch_y.astype(float)
                        
             _, l, summary_str, emb1, emb2 = sess.run([train_step, loss, merged, siamese.o1, siamese.o2],
                                         feed_dict={
                            siamese.x1: batch_x1,
                            siamese.x2: batch_x2,
                            siamese.y_: batch_y})
#             print(emb1)
    
             if np.isnan(l):
                print('Model diverged with loss = NaN')
             else:            
                 writer.add_summary(summary_str, step)
                 if step % 50 == 0:
                     print ('Epoch: %d     step: %d    loss = %.3f' % (step / steps_per_epoch, step % steps_per_epoch, l))
#                 if step % steps_per_epoch == 0:
#                     print ('############## epoch %d#############: loss %.3f' % (step / steps_per_epoch, l))
#                 print("\r#%d - Loss"%step, l)                   
                 if step % steps_per_epoch == 0:
                     saver.save(sess, save_dir + 'model')
            
    print('Training is done.')
    return 0 
