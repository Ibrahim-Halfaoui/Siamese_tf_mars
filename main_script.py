""" Siamese implementation for person Reidentification using Tensorflow 
@ author: ibrahim Halfaoui 04.2020
"""
# Imports
from __future__ import absolute_import, division, print_function
# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import argparse
import tensorflow as tf

# Scripts import
from test_net import test_net
from train_net import train_net
from dloader import dataloader

#Arguments Parsing
parser = argparse.ArgumentParser(description='Model Person Re-Identification TensorFlow implementation.')
parser.add_argument('--mode',                      type=str,   help='train or test', default='test')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='/mnt/disk4/tensorflow_projects/person_RID/dataset' ) 
parser.add_argument('--input_height',              type=int,   help='input height', default=128)
parser.add_argument('--input_width',               type=int,   help='input width', default=256)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=20) #mod here---------------------------------------------
parser.add_argument('--num_epochs',                 type=int,   help='number of epochs', default=100)
#parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
#parser.add_argument('--output_directory',          type=str,   help='output directory, if empty outputs to checkpoint folder', default='')
#parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
#parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
args = parser.parse_args()

#Main
def main(_):
    if args.mode == 'train':
        # Set up train data path 
        train_data_path = args.data_path + '/bbox_train/'
        
        # Run train function
        train_net(train_data_path, args.mode, args.batch_size, args.num_epochs)
        
    elif args.mode == 'test':
         # Set up test data path
        test_data_path = args.data_path + '/bbox_test/'
        
        # Generate Two random images for testing
        data = dataloader(test_data_path, args.mode, 1)
        x1, l1, x2, l2 = data.get_batch()
                   
        # Run Test function
        test_net(x1, x2)

if __name__ == '__main__':
    tf.app.run()
