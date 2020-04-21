""" Data loader implementation for person Re-Id using Tensorflow 
@ author: ibrahim Halfaoui
"""
import os
import tensorflow as tf
import random
import glob
import numpy as np
from PIL import Image

class dataloader(object):
    """ dataloader"""
    def __init__(self, data_path, mode, batch_size):
        # Initilization
        self.data_path = data_path
        self.mode = mode
        self.total_size = None      
        self.batch_size = batch_size
        self.height = 128
        self.width = 256
        self.dict = {}
        self.subdirs = []
        
        # Go through data and generate a dictionary for images and labels
        if not os.path.isdir(self.data_path):
            print('Problem finding the given datset in the given path')
        else:             
             dict={}
             dataset_size = 0
             subdirs = next(os.walk(self.data_path))[1] 
             self.subdirs = subdirs
             k = 0    
             for k, sd in enumerate(subdirs):                
                   path_subdir = os.path.join(self.data_path, sd)
                   os.chdir(path_subdir)
                   img_list = [f for f in glob.glob("*.jpg")]
                   dict[k] = img_list
                   dataset_size += len(img_list)
             self.dict = dict
             self.total_size = dataset_size
                  
    
    def get_batch(self):
        # Prepare generated batch out of the whole available dataset
        imgs1 = np.zeros((self.batch_size, self.width, self.height, 3),dtype=float)
        l1 = np.zeros((self.batch_size, 1),dtype=int)
        imgs2 = np.zeros((self.batch_size, self.width, self.height, 3),dtype=float)
        l2 = np.zeros((self.batch_size, 1),dtype=int)
        
        for k in range (0, self.batch_size):
             label_1 = random.randint(0, len(self.subdirs) - 1)
             path_subdir = os.path.join(self.data_path, self.subdirs[label_1])
             x1_path = path_subdir + '/'+  random.choice(self.dict[label_1])            
             if np.random.random() < 0.5:
                 label_2 = label_1 
             else:
                label_2 = self.generate_diff_class(label_1, len(self.subdirs) - 1)
             x2_path = str(os.path.join(self.data_path, self.subdirs[label_2])) + '/' + random.choice(self.dict[label_2])
             
             x1= Image.open(x1_path).convert('RGB')
             x2= Image.open(x2_path).convert('RGB')
             
             x1= x1.resize((self.height, self.width), Image.ANTIALIAS)
             x2= x2.resize((self.height, self.width), Image.ANTIALIAS)
             
             x1 = np.array(x1).astype('float') /255
             x2 = np.array(x2).astype('float') /255
            
             imgs1[k] = x1
             imgs2[k] = x2
             l1[k] = label_1
             l2[k] = label_2
                 
        return imgs1, l1, imgs2, l2
             
    def generate_diff_class(self, label_1, l):
        # Generate random int different from given number within the \
        # range of available classes
          while True:
             x = random.randint(0,l)
             if x != label_1 :
                break
          return x  

