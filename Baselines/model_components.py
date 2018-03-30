'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import os
import sys
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.layers import Reshape, Flatten, LeakyReLU, Activation
from keras_adversarial.legacy import l1l2
from sklearn.metrics import classification_report



def get_generator(G_in, output_dim, hidden_dim=100, reg=lambda: l1l2(1e-5, 1e-5)):

    x = Dense(int(hidden_dim), name="generator_h1", W_regularizer=reg())(G_in)
    x = LeakyReLU(0.2)(x)
    x = Dense(output_dim, name="generator_x_flat", W_regularizer=reg())(x)
    G_out = Activation('tanh')(x)
    # G_out = Activation('sigmoid')(x)
    G = Model(G_in, G_out)
    G.compile(loss='binary_crossentropy', optimizer='adam')
    return G, G_out

def get_discriminator(D_in, hidden_dim=50, reg=lambda: l1l2(1e-5, 1e-5)):

    x = Dense(hidden_dim * 2, name="discriminator_h1",W_regularizer=reg())(D_in)
    x = LeakyReLU(0.2)(x)
    x = Dense(hidden_dim, name="discriminator_h2",W_regularizer=reg())(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(1, name="discriminator_y", W_regularizer=reg())(x)
    D_out = Activation("sigmoid")(x)
    D = Model(D_in, D_out)
    D.compile(loss='binary_crossentropy', optimizer='sgd')
    return D, D_out

# Freeze weights in the discriminator for stacked training
def set_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable

# Build stacked GAN model
def make_gan(GAN_in, G, D):
    set_trainable(D, False)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer='adam')
    return GAN, GAN_out

# Training Procedure Definition
def sample_data_and_gen(XT, G, noise_dim=50):
    n_samples = XT.shape[0]
    s = np.arange(2*n_samples)
    np.random.shuffle(s)
    XN_noise = np.random.normal(0, 1, size=[n_samples, noise_dim])
    XN = G.predict(XN_noise)
    X = np.concatenate((XT, XN))
    y = np.ones(2*n_samples)
    y[n_samples:] = 0
    X = X[s]
    y = y[s]
    return X, y

def sample_noise(n_samples, noise_dim=50):
    X = np.random.normal(0, 1, size=[n_samples, noise_dim])
    y = np.ones(n_samples)
    return X, y

def pretrain(G, D, XT, batch_size=50):
    X, y = sample_data_and_gen(XT, G, noise_dim=50)
    set_trainable(D, True)
    D.fit(X, y, nb_epoch=1, batch_size=batch_size)

def batch_divide(X, batch_size):
    q = np.divide(X, batch_size)
    r = np.remainder(X, batch_size)
    return q, r

def train_and_test(GAN, G, D, XT, x_test, y_test, en_ae, epochs, verbose=True, v_freq=10):
    if en_ae == 1:
        XT = XT[0:7000]
        x_test, y_test = random_sampling_test_data(x_test, y_test, 3000)
        batch_size = 700
    elif en_ae == 2:
        XT = XT[0:700]
        x_test, y_test = random_sampling_test_data(x_test, y_test, 490)
        batch_size = 70
    D_fake_prob = list()
    D_real_prob = list()
    D_val_prob = list()
    fake_real_mse = list()
    f1_score_coll = list()

    # D_loss = list()
    # G_loss = list()
    # accuracy  = list()

    for epoch in range(epochs):
        X, y = sample_data_and_gen(XT, G, noise_dim=50)
        X_real = X[y == 1]
        X_fake = X[y == 0]
        d_loss = list()
        g_loss = list()
        q, r = batch_divide(X_real.shape[0], batch_size)
        for i in range(q):
            set_trainable(D, True)
            d_loss.append(D.train_on_batch(np.array(
                                            X_real[i * batch_size:(i + 1) * batch_size].tolist() +
                                            X_fake[i * batch_size:(i + 1) * batch_size].tolist()
                                            ),
                                            np.array(
                                            np.ones(batch_size).tolist() + np.zeros(batch_size).tolist()
                                            )))

            set_trainable(D, False)
            X_gan, y_gan = sample_noise(batch_size, 50)
            g_loss.append(GAN.train_on_batch(X_gan,y_gan))

        if r != 0:
            set_trainable(D, True)
            d_loss.append(D.train_on_batch(
                np.array(
                    X_real[-r:].tolist() + X_fake[-r:].tolist()
                ),
                np.array(
                    np.ones(r).tolist() + np.zeros(r).tolist()
                )))
            set_trainable(D, False)
            X_r, y_r = sample_noise(r, 50)
            g_loss.append(GAN.train_on_batch(X_r,y_r))

        fake_real_mse.append(np.mean(np.sqrt((X_real-X_fake)**2)))
        D_fake_prob.append(np.mean(D.predict(X_fake)))
        D_real_prob.append(np.mean(D.predict(X_real)))
        D_val_prob.append(np.mean(D.predict(x_test[y_test==0])))
        # D_loss.append(np.mean(d_loss))
        # G_loss.append(np.mean(g_loss))

        y_pred = (D.predict(x_test) > .5).astype(int).flatten()
        conf_mat = classification_report(y_test, y_pred, target_names=['vandal', 'benign'], digits=4)
        f1_score = float(filter(None, conf_mat.strip().split(" "))[7])
        f1_score_coll.append(f1_score)
        # print "epoch: ", epoch, "    ", filter(None, conf_mat[-50:].strip().split(" "))[-2]
        # print "epoch:%s"%epoch
        # print conf_mat
        # f1_score.append(float(filter(None, conf_mat.strip().split(" "))[7]))
        # acc = np.sum(y_pred == y_test)/float(y_pred.shape[0])
        # accuracy.append(acc)
    return D, X_fake, D_real_prob, D_fake_prob, D_val_prob, fake_real_mse, f1_score_coll


def random_sampling_test_data(x_test, y_test,n_samples=3000):
    x_test_ben = x_test[y_test == 1]
    x_test_van = x_test[y_test != 1]
    assert x_test_ben.shape[0] == x_test_van.shape[0]
    assert x_test_ben.shape[0] >= n_samples
    # s = np.arange(x_test_ben.shape[0])
    # np.random.shuffle(s)
    # s = s[:n_samples]
    x_test = np.concatenate((x_test_ben[0:n_samples], x_test_van[0:n_samples]))
    y_test = np.ones(2 * n_samples)
    y_test[n_samples:] = 0
    return x_test, y_test


def train_gan(GAN, G, D, XT, epochs, en_ae, verbose=True, v_freq=10):
    if en_ae == 1:
        XT = XT[0:7000]
        batch_size = 700
    else:
        XT = XT[0:700]
        batch_size = 70
    for epoch in range(epochs):
        X, y = sample_data_and_gen(XT, G, noise_dim=50)
        X_real = X[y == 1]
        X_fake = X[y == 0]
        d_loss = list()
        g_loss = list()
        q, r = batch_divide(X_real.shape[0], batch_size)
        for i in range(q):
            set_trainable(D, True)
            d_loss.append(D.train_on_batch(np.array(
                                            X_real[i * batch_size:(i + 1) * batch_size].tolist() +
                                            X_fake[i * batch_size:(i + 1) * batch_size].tolist()
                                            ),
                                            np.array(
                                            np.ones(batch_size).tolist() + np.zeros(batch_size).tolist()
                                            )))

            set_trainable(D, False)
            X_gan, y_gan = sample_noise(batch_size, 50)
            g_loss.append(GAN.train_on_batch(X_gan,y_gan))

        if r != 0:
            set_trainable(D, True)
            d_loss.append(D.train_on_batch(
                np.array(
                    X_real[-r:].tolist() + X_fake[-r:].tolist()
                ),
                np.array(
                    np.ones(r).tolist() + np.zeros(r).tolist()
                )))
            set_trainable(D, False)
            X_r, y_r = sample_noise(r, 50)
            g_loss.append(GAN.train_on_batch(X_r,y_r))
    return D


def run_Gan(x_test, y_test, D, en_ae):
    if en_ae == 1:
        x_test, y_test = random_sampling_test_data(x_test, y_test, 3000)
    else:
        x_test, y_test = random_sampling_test_data(x_test, y_test, 490)

    y_pred = (D.predict(x_test) > .5).astype(int).flatten()
    conf_mat = classification_report(y_test, y_pred, target_names=['vandal', 'benign'], digits=4)
    acc = np.sum(y_pred == y_test) / float(y_pred.shape[0])
    print conf_mat
    return np.array(filter(None, conf_mat.strip().split(" "))[5]).astype(float),\
           np.array(filter(None, conf_mat.strip().split(" "))[6]).astype(float), \
           np.array(filter(None, conf_mat.strip().split(" "))[7]).astype(float), acc

def run_one_svm(x_test, y_test, clf, en_ae):
    if en_ae == 1:
        N = 3000
    else:
        N = 490
    x_test_svm = np.concatenate((x_test[y_test == 1][0:N], x_test[y_test == 2][0:N]))
    y_test_svm = np.concatenate((np.ones(N), np.zeros(N)-1))
    y_pred = clf.predict(x_test_svm)
    conf_mat = classification_report(y_test_svm, y_pred, target_names=['vandal', 'benign'], digits=4)
    acc = np.sum(y_pred == y_test_svm) / float(y_pred.shape[0])

    return np.array(filter(None, conf_mat.strip().split(" "))[5]).astype(float),\
           np.array(filter(None, conf_mat.strip().split(" "))[6]).astype(float), \
           np.array(filter(None, conf_mat.strip().split(" "))[7]).astype(float), acc
