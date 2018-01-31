import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from numpy.random import multivariate_normal
from representation_libs import db_span, get_eps, cluster_analyis, DB_statistics
import json
from utils import sample_shuffle
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.spatial import distance

from matplotlib.axes import Axes

x_ben = np.load("ben_hid_repre.npy")
x_van = np.load("van_hid_repre.npy")

x_fake = sample_shuffle(np.load("x_fake.npy"))[0:len(x_van)]

X = np.concatenate((x_ben, x_van, x_fake))
y = np.concatenate((np.ones(len(x_ben)), np.zeros(len(x_van)), np.ones(len(x_fake)) + 1))
eps_X = get_eps(X)

clusters, outlier = db_span(X, 1.4305, 180)

# clusters, outlier = db_span(X, eps_X*.48, 180)
# print "eps: ", eps_X*.48



cluster_X = list()
cluster_y = list()
cluster_c = list()
for cluster_id, class_ids in clusters.items():
    cluster_X.extend(X[class_ids])
    cluster_y.extend(y[class_ids])
    cluster_c.extend((np.zeros(np.sum(class_ids))+cluster_id).tolist())
cluster_X, cluster_y, cluster_c = np.array(cluster_X), np.array(cluster_y), np.array(cluster_c)
np.save("cluster_X", cluster_X)
np.save("cluster_y", cluster_y)
np.save("cluster_c", cluster_c)

cluster_label = list()
cluster_samples = list()
for cid in set(cluster_c):
    cluster_label.append(cluster_y[cluster_c == cid])
    cluster_samples.append(cluster_X[cluster_c == cid])

for i, e in enumerate(cluster_label):
    tmp = np.array([np.sum(e == 1), np.sum(e == 0), np.sum(e == 2)])
    print "cluster %s: "%i, tmp, tmp/float(np.sum(tmp)), np.sum(tmp)

for i in np.arange(len(cluster_samples)):
    for j in np.arange(len(cluster_samples)):
        if i != j:
            inter_dist = distance.euclidean(np.mean(cluster_samples[i], axis=0),
                               np.mean(cluster_samples[j], axis=0))
            print "cluster %s & %s: %s"%(i, j, inter_dist)


print "*****************************************************************"

i += 1
print "Outlier components: "
outlier_y = y[outlier]
outlier_component = np.array([np.sum(outlier_y == 1), np.sum(outlier_y == 0), np.sum(outlier_y == 2)])
print "cluster %s: " % i, outlier_component, outlier_component / float(np.sum(outlier_component)), np.sum(outlier_component)

# X_2D = TSNE(n_components=2, random_state=0).fit_transform(cluster_X)
# X_3D = TSNE(n_components=3, random_state=0).fit_transform(cluster_X)
#
# np.save("X_2D", X_2D)
# np.save("X_3D", X_3D)
# exit(0)


#  2-dimensional and 3-dimensional visualization with cluster label
# X_2D = np.load("X_2D.npy")
# X_3D = np.load("X_3D.npy")
# cluster_c = np.load("cluster_c.npy")
#
# X_3D_c = list()
# for cid in set(cluster_c):
#     X_3D_c.append(X_3D[cluster_c == cid])
# X_3D_c = np.array(X_3D_c)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# markers = ["*", "v", "s", "o"]
# colors = ['b', 'c', 'y', 'm', 'r']
# for i, e in enumerate(X_3D_c):
#     if i == 2:
#         continue
#     ax.scatter( e[:, 1],e[:, 0], e[:, 2], marker=markers[i], color=colors[i])
#
# ax.set_xlim([-8,8])
# ax.set_ylim([-8,8])
# ax.set_zlim([-8,8])
# ax.tick_params(labelleft = False, labelbottom=False)
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
# ax.zaxis.set_visible(False)
# plt.tight_layout()
# plt.show()


# X_2D_c = list()
# for cid in set(cluster_c):
#     X_2D_c.append(X_2D[cluster_c == cid])
# X_2D_c = np.array(X_2D_c)
# fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# markers = ["*", "v", "s", "o"]
# colors = ['b', 'c', 'y', 'm', 'r']
# for i, e in enumerate(X_2D_c):
#     if i == 2:
#         continue
#     plt.scatter(e[:, 0], e[:, 1], marker=markers[i], color=colors[i])
# plt.show()
# exit(0)






# 2-dimensional visualization for hidden representation: benign users and vandals.

# X_2D_b_v = TSNE(n_components=2, random_state=0).fit_transform(np.concatenate((x_ben, x_van)))
# np.save("X_2D_b_v", X_2D_b_v)

# X_2D_b_v = np.load("X_2D_b_v.npy")
#
# fig = plt.figure()
# fig.patch.set_facecolor('w')
# ax = fig.add_subplot(111)
# ax.set_axis_off()
#
# markers = ["*", "v", "s", "o"]
# colors = ['b', 'c', 'y', 'm', 'r']
#
#
# X_2D_ben = X_2D_b_v[0:len(x_ben)]
# X_2D_van = X_2D_b_v[len(x_ben):]
#
# plt.scatter(X_2D_ben[:, 0], X_2D_ben[:, 1], marker='*', color='b')
# plt.scatter(X_2D_van[:, 0], X_2D_van[:, 1], marker='v', color='c')
#
# ax.set_xlim([-13,13])
# ax.set_ylim([-13,13])
# plt.show()