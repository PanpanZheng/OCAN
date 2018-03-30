'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import os
import sys
sys.path.append(os.getcwd() + "\\..\\..\\")
import numpy as np
from utils import sample_shuffle
from base_classifiers import svm_oneclass, elliptic_envelope, iso_forest
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

from baseline_OCC_utils import *
from base_classifiers import LSTM_Autoencoder, svm_oneclass, Dense_Autoencoder
from model_components import train_gan, run_Gan,run_one_svm
from utils import sample_shuffle, draw_trend, plot_decision_boundary, TSNE_2D_show_tri


matlab_script_path = "C:\\Users\\Panpan_user\\Documents\\MATLAB\\"
matlab_eng = matlab_engine_setup(matlab_script_path)

# try_num = sys.argv[1:]
# Load data and preprocess.
en_ae = 1  # 1 for wiki; 2 for credit card with encoding; 3 for credit card without encoding.
dra_tra_pro = True # Observe the training process along epochs, or run training then test it.

if en_ae == 1:
    samples_path = os.getcwd() + "\\..\\..\\sampleData\\"
    f_ben, f_van = "X_v8_4_50_Ben", "X_v8_4_50_Van"
    x_ben, x_van = load_data(samples_path, f_ben, f_van)
    input_dim = 8
    hid_dim = [200]
    d_in = [200]
    epochs = 150
elif en_ae == 2:
    x_ben, x_van = getDataCCFD("creditcard.csv.zip")
    x_ben = sample_shuffle(x_ben)[0:2000]
    input_dim = 30
    hid_dim = [100]
    d_in = [50]  #autoencoding.
    epochs = 200
else:
    x_ben, x_van = getDataCCFD("creditcard.csv.zip")
    x_ben = sample_shuffle(x_ben)[0:2000]
    input_dim = 30
    d_in = [input_dim]  # without autoencoding.
    epochs = 200

train_ratio = .7
max_len = 50
time_step = max_len
g_in = [50]
gan_in = [50]
sampling_ratio = train_ratio
neg_label_OCC = 2
neg_label_GAN = 0
iter_num = 10

prec_coll = list()
reca_coll = list()
f1_score_coll = list()
accuracy_coll = list()

for i in range(iter_num):
    if en_ae == 1: # LSTM-autoencoder for wiki data.
        x_train_P, x_ben_P, x_van_P, weights_P, __, __ = sampling_preprocessing_LSTM_AE(x_ben, x_van, train_ratio, max_len)
        lstm_ae = LSTM_Autoencoder(input_dim, time_step, hid_dim)
        lstm_ae.compile()
        lstm_ae.fit(x_train_P, weights_P)
        lstm_ae.get_hidden_layer_last_step()
        ben_hid_repre, van_hid_repre = map(lambda x: lstm_ae.get_hidden_representation(x), [x_ben_P, x_van_P])
        ben_hid_repre, van_hid_repre = map(lambda x: preprocess_minus_1_and_pos_1(x), [ben_hid_repre, van_hid_repre])
    elif en_ae == 2:  # Dense encoder for Credit Card data.
        dense_ae = Dense_Autoencoder(input_dim, hid_dim)
        dense_ae.compile()
        dense_ae.fit(x_ben[0:700])
        dense_ae.get_hidden_layer()
        ben_hid_repre, van_hid_repre = map(lambda x: dense_ae.get_hidden_representation(x), [x_ben, x_van])
        ben_hid_repre, van_hid_repre = map(lambda x: preprocess_minus_1_and_pos_1(x), [ben_hid_repre, van_hid_repre])
        # np.save("ben_hid_repre_r%s"%i, ben_hid_repre)
        # np.save("van_hid_repre_r%s"%i, van_hid_repre)
    else:
        ben_hid_repre, van_hid_repre = map(lambda x: preprocess_minus_1_and_pos_1(x), [x_ben, x_van])
        np.save("ben_raw_r%s"%i, ben_hid_repre)
        np.save("van_raw_r%s"%i, van_hid_repre)

    x_train, x_test, y_train_OCC, y_test_OCC, y_test_GAN  = \
        sampling_data_for_OCC(ben_hid_repre, van_hid_repre, sampling_ratio, neg_label_OCC, neg_label_GAN, en_ae)

    GAN, D, G = get_GAN(g_in, d_in, gan_in)
    if dra_tra_pro:
        D, X_fake, D_real_prob, D_fake_prob, D_val_prob, fake_real_mse, f1_score = \
            train_and_test(GAN, G, D, x_train, x_test, y_test_GAN, en_ae, epochs)
        x_test_ben = x_test[y_test_GAN == 1]
        x_test_van = x_test[y_test_GAN != 1]
        x_test_ben = sample_shuffle(x_test_ben)
        x_test_van = sample_shuffle(x_test_van)
        X_fake = sample_shuffle(X_fake)
        X = x_test_ben[0:1000].tolist() +  X_fake[0:1000].tolist() + x_test_van[0:1000].tolist()
        y = np.ones(1000).tolist() + np.zeros(1000).tolist() + (np.ones(1000)+1).tolist()
        X, y = np.array(X), np.array(y)
        TSNE_2D_show_tri(X, y)
        draw_trend(D_real_prob, D_fake_prob, D_val_prob, fake_real_mse, f1_score)
        exit(0)
    else:
        discriminator = train_gan(GAN, G, D, x_train, epochs, en_ae)
        prec_gan, reca_gan, f1_gan, acc_gan = run_Gan(x_test, y_test_GAN, discriminator, en_ae)

    prec_OCC, reca_OCC, f1_OCC, acc_OCC = run_OCC(x_train, x_test, y_train_OCC, y_test_OCC, matlab_eng, i, en_ae)
    if en_ae == 1:
        clf = svm_oneclass(x_train[0:7000])
    else:
        clf = svm_oneclass(x_train[0:700])
    prec_svm, reca_svm, f1_svm, acc_svm = run_one_svm(x_test, y_test_OCC, clf, en_ae)

    prec_coll.append([prec_gan] + prec_OCC + [prec_svm])
    reca_coll.append([reca_gan] + reca_OCC + [reca_svm])
    f1_score_coll.append([f1_gan] + f1_OCC + [f1_svm])
    accuracy_coll.append([acc_gan] + acc_OCC + [acc_svm])

prec_coll, reca_coll, f1_score_coll, accuracy_coll = \
    np.array(prec_coll), np.array(reca_coll), np.array(f1_score_coll), np.array(accuracy_coll)

print "====================== precision ================================="

print "prec_gan: ", map(lambda x: decimal_precision(x, 4), [np.mean(prec_coll[:,0]),
                                                           np.std(prec_coll[:,0])])
print "prec_gpoc: ", map(lambda x: decimal_precision(x, 4), [np.mean(prec_coll[:,1]),
                                                           np.std(prec_coll[:,1])])

print "prec_nn: ", map(lambda x: decimal_precision(x, 4), [np.mean(prec_coll[:,2]),
                                                           np.std(prec_coll[:,2])])

print "prec_scikit_svm: ", map(lambda x: decimal_precision(x, 4), [np.mean(prec_coll[:,3]),
                                                           np.std(prec_coll[:,3])])

print "====================== recall ================================="

print "reca_gan: ", map(lambda x: decimal_precision(x, 4), [np.mean(reca_coll[:,0]),
                                                           np.std(reca_coll[:,0])])
print "reca_gpoc: ", map(lambda x: decimal_precision(x, 4), [np.mean(reca_coll[:,1]),
                                                           np.std(reca_coll[:,1])])
print "reca_nn: ", map(lambda x: decimal_precision(x, 4), [np.mean(reca_coll[:,2]),
                                                           np.std(reca_coll[:,2])])
print "reca_scikit_svm: ", map(lambda x: decimal_precision(x, 4), [np.mean(reca_coll[:,3]),
                                                           np.std(reca_coll[:,3])])

print "===================== f1 score ================================"
print "f1_score_gan: ", map(lambda x: decimal_precision(x, 4), [np.mean(f1_score_coll[:,0]),
                                                                np.std(f1_score_coll[:,0])])
print "f1_score_gpoc: ", map(lambda x: decimal_precision(x, 4), [np.mean(f1_score_coll[:,1]),
                                                                np.std(f1_score_coll[:,1])])
print "f1_score_nn: ", map(lambda x: decimal_precision(x, 4), [np.mean(f1_score_coll[:,2]),
                                                                np.std(f1_score_coll[:,2])])
print "f1_scikit_svm: ", map(lambda x: decimal_precision(x, 4), [np.mean(f1_score_coll[:,3]),
                                                                np.std(f1_score_coll[:,3])])

print "====================== accuracy ================================="

print "acc_gan: ", map(lambda x: decimal_precision(x, 4), [np.mean(accuracy_coll[:,0]),
                                                           np.std(accuracy_coll[:,0])])
print "acc_gpoc: ", map(lambda x: decimal_precision(x, 4), [np.mean(accuracy_coll[:,1]),
                                                           np.std(accuracy_coll[:,1])])
print "acc_nn: ", map(lambda x: decimal_precision(x, 4), [np.mean(accuracy_coll[:,2]),
                                                           np.std(accuracy_coll[:,2])])
print "acc_scikit_svm: ", map(lambda x: decimal_precision(x, 4), [np.mean(accuracy_coll[:,3]),
                                                           np.std(accuracy_coll[:,3])])
exit(0)




