'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

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
