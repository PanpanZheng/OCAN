'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import os
import sys
sys.path.append(os.getcwd() + "\\..\\..\\")
from baseline_OCC_utils import *
from base_classifiers import LSTM_Autoencoder
from model_components import train_gan

# Load data and preprocess.
samples_path = os.getcwd() + "\\..\\..\\sampleData\\"
f_ben, f_van = "X_v8_4_50_Ben", "X_v8_4_50_Van"
x_ben, x_van = load_data(samples_path, f_ben, f_van)

train_ratio = .7
max_len = 50

# Contruct the LSTM-AE
input_dim = 8
time_step = max_len
hid_dim = [200]

sampling_ratio = train_ratio

x_train_P, x_ben_P, x_van_P, weights_P, seq_len_ben, seq_len_van = sampling_preprocessing_LSTM_AE(x_ben, x_van, train_ratio, \
                                                                                                  max_len)

lstm_ae = LSTM_Autoencoder(input_dim, time_step, hid_dim)
lstm_ae.compile()
lstm_ae.fit(x_train_P, weights_P)

test_ben_P = x_ben_P[len(x_train_P):]
test_van_P = x_van_P[0:len(test_ben_P)]

test_seq_len_ben = np.array(seq_len_ben[len(x_train_P):])
test_seq_len_van = np.array(seq_len_van[0:len(test_ben_P)])

lstm_ae.get_hidden_layer_sequence()

ben_hid_repre_P = lstm_ae.get_hidden_representation(test_ben_P)
van_hid_repre_P = lstm_ae.get_hidden_representation(test_van_P)

ben_hid_last_4 = ben_hid_repre_P[:,-4:]
van_hid_last_4 = van_hid_repre_P[:,-4:]

a = ben_hid_last_4.shape
b = van_hid_last_4.shape

print a
