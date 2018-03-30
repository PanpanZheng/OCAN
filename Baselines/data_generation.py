'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import os
import sys
import numpy as np
import glob
from libs import Dataset
from utils import getPageDict, MetaPageList,sample_shuffle
from shutil import rmtree

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from libs import Autoencoder


def gen_samples():
	"""
	extract features(f1~f7) from Wikipedia data repository released by "VEWS" (http://www.cs.umd.edu/~vs/vews), and
	construct samples used in our experiment.
	:return: data samples with variant editing length(4~50) and fixed-length(10-step).

	 f1~f7:
		f1: whether or not pi is a meta-page
		f2: if f1 is yes, whether or not pi's category is empty set.   if not, f2=0
		f3: whether or not time difference is less than 3 minutes between pi-1 and pi
		f4: whether or not pi has already been edited. (re-edit)
		f5: if f4 is yes, whether or not pi equals to pi-1. (consective re-edit). if no, f5=0
		f6: if f4 is no, whether or not pi has the common category with pi-1. if yes, f6 = 1
		f7(optional): whether or not edits are reverted. This information is
					  from Wikipedia auto-bots, such as cluebot, for bad editing revert.
	"""

	base = os.getcwd()
	dataRepo = base + "\\Dataset\\"
	if not os.path.exists(dataRepo):
		raise OSError("data repository is not available.")
	else:
		f_pages = dataRepo + "pages.tsv"
		f_users = dataRepo + "users.tsv"
		files = glob.glob(dataRepo + "*.csv")
	data = Dataset()

	rawData = base + "\\rawData\\"
	if not os.path.exists(rawData):
		os.makedirs(rawData)
		data.getRawData(files, f_users, f_pages, .7, rawData)

	sampleData = base + "\\sampleData\\"
	# if os.path.exists(sampleData):
	# 	rmtree(sampleData)
	# os.makedirs(sampleData)

	X_tim = np.load(rawData + "wikiEditSeq_0.7\\X_tim.npy")
	X_pages = np.load(rawData + "wikiEditSeq_0.7\\X_pages.npy")
	X_rev = np.load(rawData + "wikiEditSeq_0.7\\X_rev.npy")
	y = np.load(rawData + "wikiEditSeq_0.7\\y.npy")
	page2id, page2Cgr, _ = getPageDict(f_pages)

	np.save(sampleData + "MetaPageList.npy", MetaPageList(files,page2id))
	metaDict = np.load(sampleData + "MetaPageList.npy")

	# excluding 'revert' information.
	data.getSamples(X_pages, X_tim, y, metaDict, page2Cgr,
				"fix", 20, None, sampleData)
	# data.getSamples(X_pages, X_tim, y, metaDict, page2Cgr,
	# 			"var", 4, 50, sampleData)

	# including 'revert' information.
	data.getSamples(X_pages, X_tim, X_rev, y, metaDict, page2Cgr,
					"fix", 20, None, sampleData)
	# data.getSamples(X_pages, X_tim, X_rev, y, metaDict, page2Cgr,
	# 			"var", 4, 50, sampleData)
	# data.getSamples(X_pages, X_tim, X_rev, y, metaDict, page2Cgr,
	# 			"var", 1, 20, sampleData)


def gen_hid_repre(fea_dim, hid_dim, fix_or_var, step_length):

	"""
	:param fea_dim: input dimension of LSTM-AE model
	:param hid_dim: output dimension of hidden representation
	:param fix_or_var:  editing sequence is fixed-length or variant-length.
	:return: fixed-length hidden representation of editing sequence.
	"""
	base_path = os.getcwd()
	samples_path = base_path + "\\sampleData\\"
	repre_path = base_path + "\\hidden_representation\\"
	if not os.path.exists(repre_path):
		os.makedirs(repre_path)

	if fix_or_var == 1:
		# Load data
		x_ben = np.load(samples_path + "X_%s_1_20_Ben.npy" %fea_dim)
		x_van = np.load(samples_path + "X_%s_1_20_Van.npy" %fea_dim)
		# print x_ben.shape, x_van.shape
		# exit(0)
		x_ben = sample_shuffle(x_ben)[0:6000]
		x_van = sample_shuffle(x_van)[0:3000]
		train_ben = x_ben[0:3000]

		# Fit Model
		timesteps = 20
		input_dim = fea_dim

		autoencoder = Autoencoder()
		autoencoder.model('lstm', [timesteps, input_dim], hid_dim)
		autoencoder.compile()
		autoencoder.fit(train_ben, "rev")

		hidModel = Sequential()
		hidModel.add(autoencoder.model.layers[0])
		hidModel.add(autoencoder.model.layers[1])

		ben_hid_emd = hidModel.predict(x_ben)
		van_hid_emd = hidModel.predict(x_van)

		# store data
		np.save(repre_path + "ben_hid_emd_20_%s_%s" % (fea_dim, hid_dim[0]), ben_hid_emd)
		np.save(repre_path + "van_hid_emd_20_%s_%s" % (fea_dim, hid_dim[0]), van_hid_emd)

	elif fix_or_var == 0:
		if step_length == 20:
			x_ben = np.load(samples_path + "X_%s_1_20_Ben.npy" % fea_dim)
			x_van = np.load(samples_path + "X_%s_1_20_Van.npy" % fea_dim)
			x_ben = sample_shuffle(x_ben)  # 16496
			x_van = sample_shuffle(x_van)  # 17015
			# train_ben = np.concatenate((x_ben[0:10000], x_van[0:10000])) # mix samples for baseline 'latent representation.'
			train_ben = x_ben[0:10000]

			sampleWeights = list()
			for e in train_ben:
				sampleWeights.append(np.ones(len(e)))

			train_ben_P = pad_sequences(train_ben, maxlen=20, dtype='float')
			x_ben_P = pad_sequences(x_ben, maxlen=20, dtype='float')
			x_van_P = pad_sequences(x_van, maxlen=20, dtype='float')

			# decoding sequence is reversed
			sampleWeights = pad_sequences(sampleWeights, maxlen=20, dtype='float', padding='post')

			timesteps = 20
			input_dim = fea_dim
			autoencoder = Autoencoder()
			autoencoder.modelMasking('lstm', [timesteps, input_dim], hid_dim)
			autoencoder.compile('temporal')
			autoencoder.fit(train_ben_P, 'rev', sampleWeights)

			hidModel = Sequential()
			hidModel.add(autoencoder.model.layers[0])
			hidModel.add(autoencoder.model.layers[1])
			hidModel.add(autoencoder.model.layers[2])

			ben_hid_emd = hidModel.predict(x_ben_P)
			van_hid_emd = hidModel.predict(x_van_P)

			# store data
			# np.save(repre_path + "ben_hid_emd_mix_1_20_%s_%s" % (fea_dim, hid_dim[0]), ben_hid_emd)
			# np.save(repre_path + "val_hid_emd_mix_1_20_%s_%s" % (fea_dim, hid_dim[0]), van_hid_emd)

		elif step_length == 50:

			x_ben = np.load(samples_path + "X_v%s_4_50_Ben.npy" %fea_dim)
			x_van = np.load(samples_path + "X_v%s_4_50_Van.npy" %fea_dim)
			x_ben = sample_shuffle(x_ben)
			x_van = sample_shuffle(x_van)
			train_ben = x_ben[0:7000]

			sampleWeights = list()
			for e in train_ben:
				sampleWeights.append(np.ones(len(e)))

			train_ben_P = pad_sequences(train_ben, maxlen=50, dtype='float')
			x_ben_P = pad_sequences(x_ben, maxlen=50, dtype='float')
			x_van_P = pad_sequences(x_van, maxlen=50, dtype='float')

			# decoding sequence is reversed
			sampleWeights = pad_sequences(sampleWeights, maxlen=50, dtype='float', padding='post')

			timesteps = 50
			input_dim = fea_dim
			autoencoder = Autoencoder()
			autoencoder.modelMasking('lstm', [timesteps, input_dim], hid_dim)
			autoencoder.compile('temporal')
			autoencoder.fit(train_ben_P, 'rev', sampleWeights)

			hidModel = Sequential()
			hidModel.add(autoencoder.model.layers[0])
			hidModel.add(autoencoder.model.layers[1])
			hidModel.add(autoencoder.model.layers[2])

			ben_hid_emd = hidModel.predict(x_ben_P)
			van_hid_emd = hidModel.predict(x_van_P)

	return ben_hid_emd, van_hid_emd

def bw_one_and_minus_one(x):
    return ((x-min(x))/float((max(x)-min(x))))*2 - 1


