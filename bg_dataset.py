'''
    Author: Panpan Zheng
    Date created:  2/15/2018
    Python Version: 2.7
'''
import numpy as np
from bg_utils import one_hot

def load_data(x_benign, x_vandal, n_b_lab, n_v_lab, n_b_test, n_v_test, oh=True):

    # labeled data (supervised)
    x_lab_ben = x_benign[0:n_b_lab]
    x_lab_van = x_vandal[0:n_v_lab]
    x_lab = x_lab_ben.tolist() + x_lab_van.tolist()
    x_lab = np.array(x_lab)
    y_lab = np.ones(len(x_lab), dtype=np.int32)
    y_lab[len(x_lab_ben):] = 0
    if oh:
        y_lab = one_hot(y_lab, 3)


    # unlabeled data (unsupervised)
    # x_unl_ben = x_benign[len(x_lab_ben):-n_b_test]
    # x_unl_van = x_vandal[len(x_lab_van):-n_v_test]
    x_unl_ben = x_benign[len(x_lab_ben):-3*n_b_test]
    x_unl_van = x_vandal[len(x_lab_van):-3*n_v_test]
    x_unl = x_unl_ben.tolist() + x_unl_van.tolist()
    x_unl = np.array(x_unl)


    # test data.
    x_benign_test = x_benign[len(x_lab_ben) + len(x_unl_ben):]
    x_vandal_test = x_vandal[len(x_lab_van) + len(x_unl_van):]
    x_test = x_benign_test.tolist() + x_vandal_test.tolist()
    x_test = np.array(x_test)
    y_test = np.ones(len(x_test), dtype=np.int32)
    y_test[len(x_benign_test):] = 0

    return x_lab, y_lab, x_unl, x_test, y_test




def load_data_unbal(x_benign, x_vandal, n_b_lab, n_v_lab, n_b_test, n_v_test, oh=True):

    # labeled data (supervised)
    x_lab_ben = x_benign[0:n_b_lab]
    x_lab_van = x_vandal[0:n_v_lab]
    x_lab = x_lab_ben.tolist() + x_lab_van.tolist()
    x_lab = np.array(x_lab)
    y_lab = np.ones(len(x_lab), dtype=np.int32)
    y_lab[len(x_lab_ben):] = 0
    if oh:
        y_lab = one_hot(y_lab, 3)

    print x_lab_ben.shape, x_lab_van.shape


    # unlabeled data (unsupervised)
    x_unl_ben = x_benign[len(x_lab_ben):-3*n_b_test]
    x_unl_van = x_vandal[len(x_lab_van):-3*n_v_test]
    x_unl = x_unl_ben.tolist() + x_unl_van.tolist()
    x_unl = np.array(x_unl)
    print x_unl_ben.shape, x_unl_van.shape


    # test data.
    x_benign_test = x_benign[len(x_lab_ben) + len(x_unl_ben):]
    x_vandal_test = x_vandal[len(x_lab_van) + len(x_unl_van):]
    x_test = x_benign_test.tolist() + x_vandal_test.tolist()
    x_test = np.array(x_test)
    y_test = np.ones(len(x_test), dtype=np.int32)
    y_test[len(x_benign_test):] = 0
    print x_benign_test.shape, x_vandal_test.shape

    return x_lab, y_lab, x_unl, x_test, y_test
