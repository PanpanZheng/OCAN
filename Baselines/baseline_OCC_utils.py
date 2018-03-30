'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import os
import sys
sys.path.append("..\\..\\")
sys.path.append(os.getcwd() + "\\..\\..\\")


import numpy as np
from data_generation import bw_one_and_minus_one
from model_components import get_generator, get_discriminator, make_gan, train_and_test
from utils import sample_shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from base_classifiers import LSTM_Autoencoder
from keras.layers import Input
import scipy.io as sio
import matlab.engine
import pandas as pd
from sklearn import preprocessing


def draw_f1_accuracy(f1_score, accuracy, ind):
    fig = plt.figure()
    axes = plt.gca()
    plt.subplot(2, 1, 1)
    plt.plot(ind, f1_score, "ro-")
    plt.ylabel('f1_score')
    axes.set_xlim([1., 20.])
    plt.subplot(2, 1, 2)
    plt.plot(ind, accuracy, "bo-")
    plt.ylabel('accuracy')
    plt.xlabel('Round #')
    axes.set_xlim([1., 20.])
    plt.show()

def load_data(data_path, f_ben, f_van):
    data_ben = np.load(data_path + "%s.npy"%f_ben)
    data_van = np.load(data_path + "%s.npy"%f_van)
    return data_ben, data_van

def preprocess_minus_1_and_pos_1(X):
    return np.array(map(lambda x: bw_one_and_minus_one(x), X))

def sampling_preprocessing_LSTM_AE(x_ben, x_van, train_ratio, max_len):
    n_samples_train = int(x_ben.shape[0] * train_ratio)
    # x_train = sample_shuffle(x_ben)[0:n_samples_train]   # shuffle and sampling data
    x_ben = sample_shuffle(x_ben)
    x_van = sample_shuffle(x_van)
    x_train = x_ben[0:n_samples_train]
    weights = get_sample_weights(x_train)     #  contruct the sample weights for LSTM-AE output

    return seq_padding(x_train, max_len, 'pre'), \
           seq_padding(x_ben, max_len, 'pre'), \
           seq_padding(x_van, max_len, 'pre'), \
           seq_padding(weights, max_len, 'post'), \
           map(lambda x: len(x), x_ben),\
           map(lambda x: len(x), x_van)# padding sequence,
                                       # 'pre' for editting sequence
                                       # 'post' for weights sequence


def sampling_data_for_OCC(x_ben, x_van, sampling_ratio, neg_label1, neg_label2, en_ae):
    n_samples_train = int(len(x_ben) * sampling_ratio)
    if en_ae == 1:
        n_samples_test = len(x_ben) - n_samples_train
    else:
        n_samples_test = len(x_van)
    # n_samples_train = int(x_ben.shape[0] * sampling_ratio)
    # n_samples_test = x_ben.shape[0] - n_samples_train
    # assert n_samples_test <= x_van.shape[0]
    # assert n_samples_test <= len(x_van)
    x_ben, x_van = sample_shuffle(x_ben), sample_shuffle(x_van)
    x_train = x_ben[0:n_samples_train]
    x_test = x_ben[-n_samples_test:].tolist() + x_van[-n_samples_test:].tolist()
    x_test = np.array(x_test)
    y_train_OCC = np.ones(n_samples_train)
    y_test_OCC = np.ones(2 * n_samples_test)
    y_test_OCC[n_samples_test:] = neg_label1
    y_test_GAN = np.ones(2 * n_samples_test)
    y_test_GAN[n_samples_test:] = neg_label2
    return x_train, x_test, \
           y_train_OCC, y_test_OCC, y_test_GAN

def decimal_precision(x, digit_num):
    if "e" in str(x):
        x_decimal = x
    else:
        itgr_part, frac_part = str(x).split(".")
        if len(frac_part) > digit_num:
            x_decimal = itgr_part + "." + frac_part[0:digit_num]
        else:
            x_decimal = itgr_part + "." + frac_part
    return float(x_decimal)

def conf_mat_f1_accuracy(y_test, y_pred, tgt_nam1, tgt_nam2):
    conf_mat = classification_report(y_test, y_pred, target_names=[tgt_nam1, tgt_nam2], digits=4)
    f1 = float(filter(None, conf_mat[-50:].strip().split(" "))[-2])  # avarage f1 of tgt_nam1 and tgt_nam2
    acc = accuracy_score(y_test, y_pred)
    f1, acc = map(lambda x: decimal_precision(x, 4), [f1, acc])
    return conf_mat, f1, acc


def get_sample_weights(samples):
    sampleWeights = list()
    for e in samples:
        sampleWeights.append(np.ones(len(e)))
    return sampleWeights

def seq_padding(sample_sequence, max_length, padding_type):
    return pad_sequences(sample_sequence, maxlen=max_length, dtype='float', padding=padding_type)

def get_GAN(g_in, d_in, gan_in):
    G_in = Input(shape=g_in)
    G, G_out = get_generator(G_in, d_in[0])
    # discriminator (x -> y)
    D_in = Input(shape=d_in)
    D, D_out = get_discriminator(D_in)
    GAN_in = Input(shape=gan_in)
    GAN, GAN_out = make_gan(GAN_in, G, D)
    return GAN, D, G

def matlab_engine_setup(matlab_script_path):
    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_script_path, nargout=0)
    eng.addpath(matlab_script_path + "netlab3_2\\", nargout=0)
    eng.addpath(matlab_script_path + "NDtoolv0.12\\", nargout=0)
    eng.addpath(matlab_script_path + "NDtoolv0.12\\Netlab\\", nargout=0)
    return eng

def  run_OCC(x_train, x_test, y_train_OCC, y_test_OCC, eng, i, en_ae):
    # nd_type = ['gpoc', 'svmsch', 'nn', 'kpca']
    nd_type = ['gpoc', 'nn']
    mat_store_path = os.getcwd() + "\\..\\..\\hidden_representation\\mat_OCC\\"
    prec_container, reca_container, f1_container, acc_container = list(), list(), list(), list()
    X = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train_OCC, y_test_OCC))
    sio.savemat(mat_store_path + "X_hid_emd_4_50_8_200_r%s.mat"%i, dict(x=X, y=y))
    for tp in nd_type:
        prec, reca, f1, acc = eng.run_baseline(mat_store_path + "X_hid_emd_4_50_8_200_r%s.mat"%i, tp, en_ae, nargout=4)
        prec_container.append(prec)
        reca_container.append(reca)
        f1_container.append(f1)
        acc_container.append(acc)

    return prec_container, reca_container, f1_container, acc_container

def sampling_data_for_dynamic(x_ben, x_van, sampling_ratio, neg_label):
    n_samples_train = int(len(x_ben) * sampling_ratio)
    n_samples_test = len(x_ben) - n_samples_train
    assert n_samples_test <= x_van.shape[0]
    x_ben, x_van = sample_shuffle(x_ben), sample_shuffle(x_van)
    x_train = x_ben[0:n_samples_train]
    x_test = x_ben[-n_samples_test:].tolist() + x_van[-n_samples_test:].tolist()
    x_test = np.array(x_test)
    y_test_GAN = np.ones(2 * n_samples_test)
    y_test_GAN[n_samples_test:] = neg_label
    return x_train, x_test, y_test_GAN

def getDataCCFD(f_name):
    data = pd.read_csv(f_name)
    X = data.loc[: ,data.columns!='Class']
    X.loc[:,'Time'] = (X.loc[:,'Time'].values/3600)%24
    y = data.loc[:,'Class']
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X.values)
    y = y.values
    return X[y==0], X[y==1]
