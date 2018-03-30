'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import sys
sys.path.append("..\\..\\")
import os
import numpy as np
from libs import Autoencoder
from keras.models import Sequential, Model
from keras.layers import Input, LSTM
from keras.layers.core import Masking

from sklearn import tree, ensemble, neighbors, svm, covariance

def k_NN(X,y):
	clf = neighbors.KNeighborsClassifier(n_neighbors=3)
	return clf.fit(X,y)

def decision_tree(X,y):
	clf = tree.DecisionTreeClassifier()
	return clf.fit(X, y)

def random_forest(X,y):
	clf = ensemble.RandomForestClassifier(n_estimators=10)
	return clf.fit(X,y)

def svm_svc(X,y):
	clf = svm.SVC()
	return clf.fit(X,y)

def svm_nusvc(X,y):
	clf = svm.NuSVC()
	return clf.fit(X,y)

def svm_linearsvc(X,y):
	clf = svm.LinearSVC()
	return clf.fit(X,y)

def svm_oneclass(X):
	clf = svm.OneClassSVM()
	return clf.fit(X)

def elliptic_envelope(X):
	clf = covariance.EllipticEnvelope()
	return clf.fit(X)

def iso_forest(X):
	clf = ensemble.IsolationForest(max_samples=X.shape[0], random_state=None)
	return clf.fit(X)

class LSTM_Autoencoder(object):
	"""docstring for LSTM_Autoencoder"""
	def __init__(self, input_dim, time_step, hidden_dim):
		self.input_dim = input_dim
		self.time_step = time_step
		self.hidden_dim = hidden_dim
		self.autoencoder = Autoencoder()
		self.autoencoder.modelMasking('lstm', [self.time_step, self.input_dim], self.hidden_dim)

	def compile(self):
		self.autoencoder.compile('temporal')

	def fit(self, data, weights):
		self.autoencoder.fit(data, 'rev', weights)

	def get_hidden_layer_last_step(self):
		# print "net summary: ", self.autoencoder.model.summary()
		self.hidden_representation = Sequential()
		self.hidden_representation.add(self.autoencoder.model.layers[0])
		self.hidden_representation.add(self.autoencoder.model.layers[1])
		self.hidden_representation.add(self.autoencoder.model.layers[2])

	def get_hidden_layer_sequence(self):
		inputData = Input(shape=(self.time_step, self.input_dim))
		mask = Masking(mask_value=0.)(inputData)
		encoded = LSTM(self.hidden_dim[0], return_sequences=True, weights=self.autoencoder.model.layers[2].get_weights())(mask)
		self.hidden_representation = Model(inputData, encoded)

	def get_hidden_representation(self, data):
		return self.hidden_representation.predict(data)

class Dense_Autoencoder(object):
	"""docstring for LSTM_Autoencoder"""
	def __init__(self, input_dim, hidden_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.autoencoder = Autoencoder()
		self.autoencoder.modelMasking('dense', [self.input_dim], self.hidden_dim)

	def compile(self):
		self.autoencoder.compile()

	def fit(self, data):
		self.autoencoder.fit(data, 'nor')

	def get_hidden_layer(self):
		# print "net summary: ", self.autoencoder.model.summary()
		self.hidden_representation = Sequential()
		self.hidden_representation.add(self.autoencoder.model.layers[0])
		self.hidden_representation.add(self.autoencoder.model.layers[1])
		self.hidden_representation.add(self.autoencoder.model.layers[2])

	def get_hidden_representation(self, data):
		return self.hidden_representation.predict(data)




