'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

def db_span(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    outlier = None
    cluster_label = dict()
    for label_id in set(db.labels_):
        if label_id == -1:
            outlier = (db.labels_ == label_id)
            continue
        cluster_label[label_id] = (db.labels_ == label_id)
    return cluster_label, outlier

def cluster_analyis(cluster_label):
    dict_of_cluster = dict()
    for label_id in cluster_label:
        dict_of_cluster[label_id] = np.sum(cluster_label[label_id])
    return dict_of_cluster

def get_eps(X):
    X_contre = np.mean(X, axis=0)
    diff_to_contre = X - X_contre
    dist_to_contre = list(map(lambda x: np.sqrt(np.sum(x ** 2)), diff_to_contre))
    return np.mean(dist_to_contre)

def DB_statistics(X, eps, min_num_samples):
    cluster_index, outlier_index = db_span(X, eps, min_num_samples)
    dict_of_cluster = cluster_analyis(cluster_index)
    print("num_of_cluster: ", len(dict_of_cluster.keys()))
    print("cluster_list ", dict_of_cluster)
    print("outlier_rate: ", np.sum(outlier_index), np.sum(outlier_index)/ float(len(outlier_index)))
