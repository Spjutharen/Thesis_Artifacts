import h5py
from detect_adv import *
from utils import *
import sys
sys.path.append('../Thesis_CNN_mnist/')
from cnn import MnistCNN
sys.path.append('../Thesis_Utilities/')
from utilities import load_datasets

import tensorflow as tf
import numpy as np


# Load MNIST and, if omniglot_bool, Omniglot datasets.
x_train, y_train, _, _, x_test, y_test = load_datasets(test_size=10000, val_size=40000, omniglot_bool=True,
                                                               name_data_set='data_omni.h5', force=False,
                                                               create_file=True, r_seed=None)

# Build model.
tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/MnistCNN_save/')

# Compute kernel density estimations and linear regression model.
kdes, lr = create_detector(net, x_train, y_train, x_test, y_test, dataset='mnist')

print('DONE')
