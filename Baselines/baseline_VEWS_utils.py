'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import os
import sys
sys.path.append(os.getcwd() + "\\..\\OCC\\")
sys.path.append(os.getcwd() + "\\..\\..\\")
import numpy as np
from utils import sample_shuffle
from baseline_OCC_utils import get_sample_weights, seq_padding, bw_one_and_minus_one

def sample_shuffle_with_label(X,y):
	n_samples = X.shape[0]
	s = np.arange(n_samples)
	np.random.shuffle(s)
	return X[s], y[s]

def sampling_preprocessing_LSTM_AE(x_ben, x_van, train_ratio, max_len):

    assert train_ratio < 1.
    n_samples_train = int(x_ben.shape[0] * train_ratio)

    assert n_samples_train <= x_van.shape[0]
    x_train = sample_shuffle(x_ben)[0:n_samples_train].tolist() + \
              sample_shuffle(x_van)[0:n_samples_train].tolist()
    x_train = sample_shuffle(np.array(x_train))
    weights = get_sample_weights(x_train)     #  contruct the sample weights for LSTM-AE output.
    return seq_padding(x_train, max_len, 'pre'), \
           seq_padding(x_ben, max_len, 'pre'), \
           seq_padding(x_van, max_len, 'pre'), \
           seq_padding(weights, max_len, 'post') # 'post' for weights sequence

def sampling_data_for_VEWS(x_ben, x_van):

    y_ben, y_van = np.ones(x_ben.shape[0]), np.zeros(x_van.shape[0])
    x_ben, y_ben = sample_shuffle_with_label(x_ben, y_ben)
    x_van, y_van = sample_shuffle_with_label(x_van, y_van)
    return x_ben, x_van, y_ben, y_van

def k_fold_indices(n_samples, i, step):
	indices = np.arange(n_samples)
	test_indices = xrange(i * step, (i + 1) * step)
	train_indices = np.setdiff1d(indices, test_indices)
	return test_indices, train_indices
