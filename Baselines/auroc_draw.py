import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.preprocessing import MinMaxScaler
from bg_utils import sample_shuffle_uspv
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn import metrics
from sklearn.metrics import classification_report


# Create data
# N = 60
# g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N), 0.4 + 0.1 * np.random.rand(N))
# g2 = (0.4 + 0.3 * np.random.rand(N), 0.5 * np.random.rand(N), 0.1 * np.random.rand(N))
# g3 = (0.3 * np.random.rand(N), 0.3 * np.random.rand(N), 0.3 * np.random.rand(N))


def draw_3D(X,y):

    colors = ("blue", "magenta", "cyan")
    groups = ("Benign", "Fake", "Vandal")
    markers = ("*", "o", "v")
    # Create plot


    # fig = plt.figure()
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0", projection='3d')

    for i in range(3):
        ax.scatter(X[y == i][:, 0], X[y == i][:, 1], X[y == i][:, 2], marker=markers[i], alpha=0.8, c=colors[i], edgecolors='face', s=5,
                   label=groups[i])

    # plt.axis('off')
    plt.title('Matplot 3d scatter plot')
    plt.legend(loc=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # ax.set_zticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    plt.show()

def draw_2D(X, y):

    colors = ("blue", "c")
    groups = ("Benign", "Vandal")
    markers = ("*", "v")

    # Create plot

    # fig = plt.figure()
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

    for i in range(2):
        ax.scatter(X[y == i][:, 0], X[y == i][:, 1], marker=markers[i], alpha=0.8, c=colors[i], edgecolors='face', s=5, label=groups[i])

    plt.axis('off')
    # plt.title('Matplot 3d scatter plot')
    # plt.legend(loc=2)
    # ax.set_zticklabels([])
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    plt.show()

def roc_curve(y, pred, title):

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)'%auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s'%title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def roc_curve_two(y, pred, y2, pred2, title):

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)

    fpr2, tpr2, thresholds2 = metrics.roc_curve(y2, pred2, pos_label=1)
    auc_val2 = metrics.auc(fpr2, tpr2)


    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='c',
         lw=lw, label='representation (area = %0.4f)'%auc_val)

    plt.plot(fpr2, tpr2, color='darkorange', linestyle=":",
         lw=lw, label='raw feature (area = %0.4f)'%auc_val2)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s'%title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()



#y_test_wiki = np.load("y_test_wiki.npy")[0:3300]
#y_pred_wiki = np.load("y_prob_wiki.npy")[0:3300,1]


y_test_credit = np.load("y_test_credit.npy")[0:1100]
y_pred_credit = np.load("y_prob_credit.npy")[0:1100,1]

y_test_credit_noencoding = np.load("y_test_credit_noencoding.npy")[0:1100]
y_pred_credit_noencoding = np.load("y_prob_credit_noencoding.npy")[0:1100,1]

roc_curve_two(y_test_credit, y_pred_credit, y_test_credit_noencoding, y_pred_credit_noencoding, "")

#roc_curve(y_test_wiki, y_pred_wiki, "")
#roc_curve(y_test_credit, y_pred_credit, "")

# y_test_wiki = np.load("y_test_wiki.npy")[0:3300]
# y_pred_wiki = (np.load("y_prob_wiki.npy")[0:3300,1] > 0.5).astype(int)

# y_test_credit = np.load("y_test_credit.npy")[0:1300]
# y_pred_credit = (np.load("y_prob_credit.npy")[0:1300,1] > 0.5).astype(int)




# conf_mat_wiki = classification_report(y_test_wiki, y_pred_wiki, target_names=['benign', 'vandal'], digits=4)
# conf_mat_cred = classification_report(y_test_credit, y_pred_credit, target_names=['benign', 'vandal'], digits=4)

# print conf_mat_wiki
#print conf_mat_cred


exit(0)


min_max_scaler = MinMaxScaler()
x_benign = min_max_scaler.fit_transform(np.load("./hidden_repre/ben_hid_emd_4_50_8_200_r0.npy"))
x_vandal = min_max_scaler.fit_transform(np.load("./hidden_repre/val_hid_emd_4_50_8_200_r0.npy"))

x_benign = sample_shuffle_uspv(x_benign)
x_vandal = sample_shuffle_uspv(x_vandal)

X = x_benign[0:3000].tolist() + x_vandal[0:3000].tolist()
y = np.zeros(3000).tolist() + np.ones(3000).tolist()
X, y = np.array(X), np.array(y)

model_2D = Isomap(n_components=2)
X_2D = model_2D.fit_transform(X)


draw_2D(X_2D, y)



exit(0)























min_max_scaler = MinMaxScaler()

# if en_ae == 1:
#     x_benign = min_max_scaler.fit_transform(np.load("./hidden_repre/ben_hid_emd_4_50_8_200_r0.npy"))
#     x_vandal = min_max_scaler.transform(np.load("./hidden_repre/val_hid_emd_4_50_8_200_r0.npy"))
# elif en_ae == 2:
#     x_benign = min_max_scaler.fit_transform(np.load("./hidden_repre/credit_card/ben_hid_repre_r2.npy"))
#     x_vandal = min_max_scaler.transform(np.load("./hidden_repre/credit_card/van_hid_repre_r2.npy"))
# else:
#     x_benign = min_max_scaler.fit_transform(np.load("./raw_credit_card/ben_raw_r0.npy"))
#     x_vandal = min_max_scaler.transform(np.load("./raw_credit_card/van_raw_r0.npy"))


#x_benign = min_max_scaler.fit_transform(np.load("./hidden_output/ben_hid_emd_4_50_8_200.npy"))
#x_vandal = min_max_scaler.transform(np.load("./hidden_output/val_hid_emd_4_50_8_200.npy"))


def gen_circle_data(num_samples=11000):

    # make a simple unit circle
    theta = np.linspace(0, 2*np.pi, num_samples)
    a, b = 1 * np.cos(theta), 1 * np.sin(theta)
    r = np.random.rand((num_samples))
    x, y = r * np.cos(theta), r * np.sin(theta)

    real_data = list()
    for i, e in enumerate(y):
        real_data.append([x[i], e])
    return np.array(real_data)


x_benign = gen_circle_data()
x_benign = sample_shuffle_uspv(x_benign)
# x_vandal = sample_shuffle_uspv(x_vandal)

x_benign = x_benign[0:10000]
x_pre = x_benign[0:7000]

# exit(0)
# print x_benign.shape, x_pre.shape
# exit(0)


# if en_ae == 1:
#     x_benign = x_benign[0:10000]
#     # x_vandal = x_vandal[0:10000]
#     x_pre = x_benign[0:7000]
# else:
#     x_pre = x_benign[0:700]

y_pre = np.zeros(len(x_pre))
y_pre = one_hot(y_pre, 2)

x_train = x_pre

y_real_mb = one_hot(np.zeros(mb_size), 2)
y_fake_mb = one_hot(np.ones(mb_size), 2)

# if en_ae == 1:
#     x_test = x_benign[-3000:].tolist() + x_vandal[-3000:].tolist()
# else:
#     x_test = x_benign[-490:].tolist() + x_vandal[-490:].tolist()
# x_test = np.array(x_test)


# y_test = np.zeros(len(x_test))
# if en_ae == 1:
#     y_test[3000:] = 1
# else:
#     y_test[490:] = 1


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

n_round = 200

# if en_ae == 1:
#     n_round = 50
# else:
#     n_round = 200


# plt.scatter(x_train[0:2000,0], x_train[0:2000,1], c="r")
#
# plt.ylim([-1.5,1.5])
# plt.xlim([-1.5,1.5])
# plt.show()
# exit(0)

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

    # if en_ae == 1:
    #     D_prob_vandal_ = sess.run(D_prob_real,
    #                               feed_dict={X_oc: x_vandal[0:7000]})
    #                               # feed_dict={X_oc:x_vandal[-490:]})
    # else:
    #     D_prob_vandal_ = sess.run(D_prob_real,
    #                               #feed_dict={X_oc: x_vandal[0:7000]})
    #                               feed_dict={X_oc:x_vandal[-490:]})

    d_ben_pro.append(np.mean(D_prob_real_[:, 0]))
    d_fake_pro.append(np.mean(D_prob_gen_[:, 0]))
    # d_val_pro.append(np.mean(D_prob_vandal_[:, 0]))
    fm_loss_coll.append(fm_loss_curr)
    print "epoch %s"%n_epoch, np.mean(fm_loss_coll)





bg_gen = sess.run([G_sample],
                  feed_dict={Z:sample_Z(2000, Z_dim)})


plt.scatter(bg_gen[:,0], bg_gen[:,1], c="r")
plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.show()

    # prob, _ = sess.run([D_prob_real, D_logit_real], feed_dict={X_oc: x_test})
    # y_pred = np.argmax(prob, axis=1)
    # conf_mat = classification_report(y_test, y_pred, target_names=['benign', 'vandal'], digits=4)
    # f1_score.append(float(filter(None, conf_mat.strip().split(" "))[12]))
    # print conf_mat

# if not dra_tra_pro:
#     acc = np.sum(y_pred == y_test)/float(len(y_pred))
#     print conf_mat
#     print "acc:%s"%acc
#
# if dra_tra_pro:
#     draw_trend(d_ben_pro, d_fake_pro, d_val_pro, fm_loss_coll, f1_score)

exit(0)








































