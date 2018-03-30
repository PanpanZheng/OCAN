'''
    Author: Panpan Zheng
    Date created:  2/15/2018
    Python Version: 2.7
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import os

from bg_utils import pull_away_loss, one_hot, xavier_init, sample_shuffle_spv, sample_shuffle_uspv, sample_Z, draw_trend
from bg_dataset import load_data, load_data_unbal
import sys



en_ae = int(sys.argv[1])  # en_ae == 1 for wiki dataset with autoencoding;
                          # en_ae == 2 for credit card dataset with autoencoding;
                          # en_ae == 3 for credit card dataset without autoencoding.

dra_tra_pro = int(sys.argv[2])  # dra_tra_pro == 1 for printing training trend, discr_probabiltiy, f1 and fm_loss;
                                # dra_tra_pro == 1 for printing training trend, discr_probabiltiy, f1 and fm_loss;


# print en_ae, dra_tra_pro
#
# exit(0)
# en_ae = 3 # 1 for wiki dataset with autoencoding; 2 for credit card dataset with autoencoding; 3 for credit card dataset without autoencoding.
# dra_tra_pro = False

if en_ae == 1:
    mb_size = 100
    dim_input = 200
elif en_ae == 2:
    mb_size = 70
    dim_input = 50
else:
    mb_size = 70
    dim_input = 30


D_dim = [dim_input, 100, 50, 2]
G_dim = [50, 100, dim_input]
Z_dim = G_dim[0]


# define placeholders for labeled-data, unlabeled-data, noise-data and target-data.

X_oc = tf.placeholder(tf.float32, shape=[None, dim_input])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
X_tar = tf.placeholder(tf.float32, shape=[None, dim_input])
# X_val = tf.placeholder(tf.float32, shape=[None, dim_input])


# declare weights and biases of discriminator.

D_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))
D_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))

D_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))
D_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))

D_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))
D_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]



# declare weights and biases of generator.

G_W1 = tf.Variable(xavier_init([G_dim[0], G_dim[1]]))
G_b1 = tf.Variable(tf.zeros(shape=[G_dim[1]]))

G_W2 = tf.Variable(xavier_init([G_dim[1], G_dim[2]]))
G_b2 = tf.Variable(tf.zeros(shape=[G_dim[2]]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


# declare weights and biases of pre-train net for density estimation.

T_W1 = tf.Variable(xavier_init([D_dim[0], D_dim[1]]))
T_b1 = tf.Variable(tf.zeros(shape=[D_dim[1]]))

T_W2 = tf.Variable(xavier_init([D_dim[1], D_dim[2]]))
T_b2 = tf.Variable(tf.zeros(shape=[D_dim[2]]))

T_W3 = tf.Variable(xavier_init([D_dim[2], D_dim[3]]))
T_b3 = tf.Variable(tf.zeros(shape=[D_dim[3]]))

theta_T = [T_W1, T_W2, T_W3, T_b1, T_b2, T_b3]


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_logit = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    return G_logit


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.softmax(D_logit)
    return D_prob, D_logit, D_h2


# pre-train net for density estimation.

def discriminator_tar(x):
    T_h1 = tf.nn.relu(tf.matmul(x, T_W1) + T_b1)
    T_h2 = tf.nn.relu(tf.matmul(T_h1, T_W2) + T_b2)
    T_logit = tf.matmul(T_h2, T_W3) + T_b3
    T_prob = tf.nn.softmax(T_logit)
    return T_prob, T_logit, T_h2


D_prob_real, D_logit_real, D_h2_real = discriminator(X_oc)

G_sample = generator(Z)
D_prob_gen, D_logit_gen, D_h2_gen = discriminator(G_sample)

D_prob_tar, D_logit_tar, D_h2_tar = discriminator_tar(X_tar)
D_prob_tar_gen, D_logit_tar_gen, D_h2_tar_gen = discriminator_tar(G_sample)
# D_prob_val, _, D_h1_val = discriminator(X_val)




# disc. loss
y_real= tf.placeholder(tf.int32, shape=[None, D_dim[3]])
y_gen = tf.placeholder(tf.int32, shape=[None, D_dim[3]])

D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_real,labels=y_real))
D_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_gen, labels=y_gen))

ent_real_loss = -tf.reduce_mean(
                        tf.reduce_sum(
                            tf.multiply(D_prob_real, tf.log(D_prob_real)), 1
                        )
                    )

ent_gen_loss = -tf.reduce_mean(
                        tf.reduce_sum(
                            tf.multiply(D_prob_gen, tf.log(D_prob_gen)), 1
                        )
                    )

D_loss = D_loss_real + D_loss_gen + 1.85 * ent_real_loss


# gene. loss
pt_loss = pull_away_loss(D_h2_tar_gen)

y_tar= tf.placeholder(tf.int32, shape=[None, D_dim[3]])
T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_logit_tar, labels=y_tar))
tar_thrld = tf.divide(tf.reduce_max(D_prob_tar_gen[:,-1]) +
                      tf.reduce_min(D_prob_tar_gen[:,-1]), 2)

# tar_thrld = tf.reduce_mean(D_prob_tar_gen[:,-1])


indicator = tf.sign(
              tf.subtract(D_prob_tar_gen[:,-1],
                          tar_thrld))
condition = tf.greater(tf.zeros_like(indicator), indicator)
mask_tar = tf.where(condition, tf.zeros_like(indicator), indicator)
G_ent_loss = tf.reduce_mean(tf.multiply(tf.log(D_prob_tar_gen[:,-1]), mask_tar))
# G_ent_loss = tf.reduce_mean(tf.log(D_prob_tar_gen[:,-1]))

fm_loss = tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(
                    tf.square(D_logit_real - D_logit_gen), 1
                    )
                )
            )

G_loss = pt_loss + G_ent_loss + fm_loss

D_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
T_solver = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(T_loss, var_list=theta_T)


# Load data....


min_max_scaler = MinMaxScaler()

if en_ae == 1:
    x_benign = min_max_scaler.fit_transform(np.load("./data/wiki/ben_hid_emd_4_50_8_200_r0.npy"))
    x_vandal = min_max_scaler.transform(np.load("./data/wiki/val_hid_emd_4_50_8_200_r0.npy"))
elif en_ae == 2:
    x_benign = min_max_scaler.fit_transform(np.load("./data/credit_card/ben_hid_repre_r2.npy"))
    x_vandal = min_max_scaler.transform(np.load("./data/credit_card/van_hid_repre_r2.npy"))
else:
    x_benign = min_max_scaler.fit_transform(np.load("./data/raw_credit_card/ben_raw_r0.npy"))
    x_vandal = min_max_scaler.transform(np.load("./data/raw_credit_card/van_raw_r0.npy"))


#x_benign = min_max_scaler.fit_transform(np.load("./hidden_output/ben_hid_emd_4_50_8_200.npy"))
#x_vandal = min_max_scaler.transform(np.load("./hidden_output/val_hid_emd_4_50_8_200.npy"))


x_benign = sample_shuffle_uspv(x_benign)
x_vandal = sample_shuffle_uspv(x_vandal)

if en_ae == 1:
    x_benign = x_benign[0:10000]
    x_vandal = x_vandal[0:10000]
    x_pre = x_benign[0:7000]
else:
    x_pre = x_benign[0:700]

y_pre = np.zeros(len(x_pre))
y_pre = one_hot(y_pre, 2)

x_train = x_pre

y_real_mb = one_hot(np.zeros(mb_size), 2)
y_fake_mb = one_hot(np.ones(mb_size), 2)

if en_ae == 1:
    x_test = x_benign[-3000:].tolist() + x_vandal[-3000:].tolist()
else:
    x_test = x_benign[-490:].tolist() + x_vandal[-490:].tolist()
x_test = np.array(x_test)


y_test = np.zeros(len(x_test))
if en_ae == 1:
    y_test[3000:] = 1
else:
    y_test[490:] = 1


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pre-training for target distribution

_ = sess.run(T_solver,
             feed_dict={
                X_tar:x_pre,
                y_tar:y_pre
                })

q = np.divide(len(x_train), mb_size)

# n_epoch = 1
#
# while n_epoch:

d_ben_pro, d_fake_pro, fm_loss_coll = list(), list(), list()
f1_score  = list()
d_val_pro = list()

if en_ae == 1:
    n_round = 50
else:
    n_round = 200

for n_epoch in range(n_round):

    X_mb_oc = sample_shuffle_uspv(x_train)

    for n_batch in range(q):

        _, D_loss_curr, ent_real_curr = sess.run([D_solver, D_loss, ent_real_loss],
                                          feed_dict={
                                                     X_oc: X_mb_oc[n_batch*mb_size:(n_batch+1)*mb_size],
                                                     Z: sample_Z(mb_size, Z_dim),
                                                     y_real: y_real_mb,
                                                     y_gen: y_fake_mb
                                                     })

        _, G_loss_curr, fm_loss_curr = sess.run([G_solver, G_loss, fm_loss],
        # _, G_loss_curr, fm_loss_, kld_ = sess.run([G_solver, G_loss, fm_loss, pt_loss + G_ent_loss],
                                           feed_dict={Z: sample_Z(mb_size, Z_dim),
                                                      X_oc: X_mb_oc[n_batch*mb_size:(n_batch+1)*mb_size],
                                                      })

    D_prob_real_, D_prob_gen_ = sess.run([D_prob_real, D_prob_gen],
                                         feed_dict={X_oc: x_train,
                                                    Z: sample_Z(len(x_train), Z_dim)})

    if en_ae == 1:
        D_prob_vandal_ = sess.run(D_prob_real,
                                  feed_dict={X_oc: x_vandal[0:7000]})
                                  # feed_dict={X_oc:x_vandal[-490:]})
    else:
        D_prob_vandal_ = sess.run(D_prob_real,
                                  #feed_dict={X_oc: x_vandal[0:7000]})
                                  feed_dict={X_oc:x_vandal[-490:]})

    d_ben_pro.append(np.mean(D_prob_real_[:, 0]))
    d_fake_pro.append(np.mean(D_prob_gen_[:, 0]))
    d_val_pro.append(np.mean(D_prob_vandal_[:, 0]))
    fm_loss_coll.append(fm_loss_curr)

    prob, _ = sess.run([D_prob_real, D_logit_real], feed_dict={X_oc: x_test})
    y_pred = np.argmax(prob, axis=1)
    conf_mat = classification_report(y_test, y_pred, target_names=['benign', 'vandal'], digits=4)
    f1_score.append(float(filter(None, conf_mat.strip().split(" "))[12]))
    # print conf_mat

if not dra_tra_pro:
    acc = np.sum(y_pred == y_test)/float(len(y_pred))
    print conf_mat
    print "acc:%s"%acc

if dra_tra_pro:
    draw_trend(d_ben_pro, d_fake_pro, d_val_pro, fm_loss_coll, f1_score)

exit(0)
