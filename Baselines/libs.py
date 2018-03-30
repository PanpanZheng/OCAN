'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import os
import sys
sys.path.append(".")
import numpy as np
from shutil import rmtree
from sklearn import preprocessing as pp
from utils import getUserDict, getPageDict, TleRevTim, TimeDiff, IsMetaPage, encode, train_test_split

from keras.layers import Input, Dense, LSTM, RepeatVector, Embedding
from keras.models import Model, Sequential
from keras.layers.core import Activation, Dense, Masking
import theano.tensor as T
from keras.callbacks import EarlyStopping
from keras import regularizers




class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self):
		super(Dataset, self).__init__()

	# def getSamples(self, X_pages, X_tim, X_rev, y, metaDict, page2Cgr, seqType, seqLenLow, seqLenUp, storePath):
	def getSamples(self, *args):

		if len(args) == 10: 
			self.X_pages = args[0]
			self.X_tim = args[1]
			self.X_rev = args[2]
			self.y = args[3]
			self.metaDict = args[4]
			self.page2Cgr = args[5]
			self.seqType = args[6]
			self.seqLenLow = args[7]
			self.seqLenUp = args[8]
			self.storePath = args[9]
			self.flag = 1
		elif len(args) == 9:
			self.X_pages = args[0]
			self.X_tim = args[1]
			self.y = args[2]
			self.metaDict = args[3]
			self.page2Cgr = args[4]
			self.seqType = args[5]
			self.seqLenLow = args[6]
			self.seqLenUp = args[7]
			self.storePath = args[8]
			self.flag = 0

		self.X = list()
		# print self.flag
		# exit(0)
		for i, pages in enumerate(self.X_pages):

			tims = self.X_tim[i]
			if self.flag: 
				revs = self.X_rev[i]
			isMetaTem = list()
			timDiffTem1 = list()
			timDiffTem2 = list()
			timDiffTem3 = list()
			reEditTem = list()
			consEditTem = list()
			comCgrTem = list()
			metEmptyTem = list()
			for j, page in enumerate(pages):
				# meta-page ? 
				isMetaTem.append(IsMetaPage(page, self.metaDict)) 
				# meta-page is empty ? 
				if IsMetaPage(page, self.metaDict): 
					if not eval(self.page2Cgr[page]):
						metEmptyTem.append(1)
					else:
						metEmptyTem.append(0)
				else:
					metEmptyTem.append(0)
			
				if j == 0:
					timDiffTem1.append(0)
					timDiffTem2.append(0)
					timDiffTem3.append(0)
					reEditTem.append(0)
					consEditTem.append(0)
					comCgrTem.append(0)
				else:
					# time difference < 1, 3, 15 mins ?
					timDiffTem1.append(TimeDiff(tims[j-1], tims[j], 1))
					timDiffTem2.append(TimeDiff(tims[j-1], tims[j], 3))
					timDiffTem3.append(TimeDiff(tims[j-1], tims[j], 15))
					# has it been edited before?
					if page in pages[0:j]:
						reEditTem.append(1)
						# Is it a consecutive edit ?
						if page == pages[j-1]:
							consEditTem.append(1)
						else: 
							consEditTem.append(0)
					else:
						reEditTem.append(0)
						consEditTem.append(0)
					# share common category ? 
					if eval(self.page2Cgr[page]).intersection(
						eval(self.page2Cgr[pages[j-1]])):
						comCgrTem.append(1)
					else:
						comCgrTem.append(0)	

			X_idvl = list()

			if self.flag: 
				for j, e in enumerate(isMetaTem):
					X_code = list()
					attrs = [e,
					timDiffTem1[j],
					timDiffTem2[j],
					timDiffTem3[j],
					reEditTem[j],
					consEditTem[j],
					comCgrTem[j],
					metEmptyTem[j],
					revs[j]
					]
					for attr in attrs: 
						X_code.extend(encode(attr, 2))
					X_idvl.append(X_code)
			else:
				for j, e in enumerate(isMetaTem):
					X_code = list()
					attrs = [e,
					timDiffTem1[j],
				 	timDiffTem2[j],
				 	timDiffTem3[j],
					reEditTem[j],
					consEditTem[j],
					comCgrTem[j],
					metEmptyTem[j]
					]
					for attr in attrs: 
						X_code.extend(encode(attr, 2))
					X_idvl.append(X_code) 

			self.X.append(pp.normalize(X_idvl, axis=1))

		self.X = np.array(self.X)
		X_value = list()
		y_value = list()

		if self.seqType == 'fix':
			for i, e in enumerate(self.X): 
				if  self.seqLenLow >= len(e):
					X_value.append(e)
				else:
					X_value.append(e[0:self.seqLenLow])
				y_value.append(self.y[i])
		elif self.seqType == 'var': 
			for i, e in enumerate(self.X): 
				if self.seqLenUp >= len(e) >= self.seqLenLow:
					X_value.append(e)
					y_value.append(self.y[i])
			# X_val_P = pad_sequences(X_val, maxlen=self.seqLenUp, dtype='float')

		X_value = np.array(X_value)
		y_value = np.array(y_value)

		# print X_val.shape
		# print y_val.shape

		x_benign = [X_value[i] for i, e in enumerate(y_value) if e == 0]
		x_vandal = [X_value[i] for i, e in enumerate(y_value) if e == 1]

		x_benign, x_vandal = np.array(x_benign), np.array(x_vandal)

		if self.seqType == 'fix':
			if self.flag:
				np.save(self.storePath + "X_18_1_20_Ben.npy", x_benign)
				np.save(self.storePath + "X_18_1_20_Van.npy", x_vandal)
			else:
				np.save(self.storePath + "X_16_1_20_Ben.npy", x_benign)
				np.save(self.storePath + "X_16_1_20_Van.npy", x_vandal)
		elif self.seqType == 'var':
			if self.flag: 
				np.save(self.storePath + "X_v8_%s_%s_Ben.npy"%(self.seqLenLow,self.seqLenUp), x_benign)
				np.save(self.storePath + "X_v8_%s_%s_Van.npy"%(self.seqLenLow,self.seqLenUp), x_vandal)
			else:
				np.save(self.storePath + "X_v6_%s_%s_Ben.npy"%(self.seqLenLow,self.seqLenUp), x_benign)
				np.save(self.storePath + "X_v6_%s_%s_Van.npy"%(self.seqLenLow,self.seqLenUp), x_vandal)



	def getRawData(self, files, f_users, f_pages, splRatio, basePath):

		self.files = files
		self.splRatio = splRatio
		self.basePath = basePath
		
		directory = self.basePath + "wikiEditSeq" + "_" + str(splRatio)
		if os.path.exists(directory):
			rmtree(directory)
		os.makedirs(directory)
		directory += "\\"

		# load user-page information from dictionary file and raw data. 
		user2id, user2Label, id2user = getUserDict(f_users)
		page2id, __, __ = getPageDict(f_pages)
		titleSet, revSet, timSet = TleRevTim(self.files,user2id,page2id)

		# user2Label = getLabel(filesB,filesV,user2id,0,1)
		
		X_usrs = list()
		y = list()
		X_pages = list()
		X_tim = list()
		X_rev = list()


		for usrid in titleSet:   # to keep userid, label, pageid, revert, editing-time consistent.  
			X_usrs.append(usrid)
			y.append(user2Label[usrid])
			X_pages.append(titleSet[usrid])
			X_rev.append(revSet[usrid])
			X_tim.append(timSet[usrid])

		X_len = [len(x) for x in X_pages]

		X_usrs, y, X_pages, X_rev, X_tim, X_len = np.array(X_usrs), np.array(y), \
						np.array(X_pages), np.array(X_rev), np.array(X_tim), np.array(X_len)


		X_usrs_train, y_train, X_pages_train, X_tim_train, X_rev_train, X_len_train, \
		X_usrs_test, y_test, X_pages_test, X_tim_test, X_rev_test, X_len_test = \
		train_test_split(X_usrs, y, X_pages, X_tim, X_rev, X_len, splRatio)

		np.save(directory + "X_usrs.npy", X_usrs)
		np.save(directory + "y.npy", y)
		np.save(directory + "X_pages.npy", X_pages)
		np.save(directory + "X_rev.npy", X_rev)
		np.save(directory + "X_tim.npy", X_tim)
		np.save(directory + "X_len.npy", X_len)

		np.save(directory + "X_usrs_train.npy", X_usrs_train)
		np.save(directory + "y_train.npy", y_train)
		np.save(directory + "X_pages_train.npy", X_pages_train)
		np.save(directory + "X_tim_train.npy", X_tim_train)
		np.save(directory + "X_rev_train.npy", X_rev_train)
		np.save(directory + "X_len_train.npy", X_len_train)

		np.save(directory + "X_usrs_test.npy", X_usrs_test)
		np.save(directory + "y_test.npy", y_test)
		np.save(directory + "X_pages_test.npy", X_pages_test)
		np.save(directory + "X_tim_test.npy", X_tim_test)
		np.save(directory + "X_rev_test.npy", X_rev_test)
		np.save(directory + "X_len_test.npy", X_len_test)



class Autoencoder(object):
	"""docstring for Autoencoder"""
	# def __init__(self, sampleWeights, sample_weight_mode):
	def __init__(self):
		# super(Autoencoder, self).__init__()
		# self.codeLayerType = 'dense'
		self.nb_epoch = 20
		self.batch_size = 256
		self.shuffle = True
		self.validation_split = 0.05
		self.optimizer = 'adadelta'
		self.loss = 'mse'
		# self.sampleWeights = sampleWeights
		# self.sample_weight_mode = sample_weight_mode


	def model(self, codeLayerType, inputDim, codeDim):

		self.codeLayerType = codeLayerType
		assert len(codeDim) > 0

		if self.codeLayerType == 'lstm':
			assert len(inputDim) == 2
			inputData = Input(shape=(inputDim[0],inputDim[1]))

			if len(codeDim) == 1:
				encoded = LSTM(codeDim[0])(inputData)
				decoded = RepeatVector(inputDim[0])(encoded)
			elif len(codeDim) > 1:
				encoded = inputData
				for i, units in enumerate(codeDim):
					if i == len(codeDim) - 1:
						 encoded = LSTM(units)(encoded)
						 continue		
					encoded = LSTM(units, return_sequences=True)(encoded)

				for i, units in enumerate(reversed(codeDim)): 
					if i == 1:
						decoded = LSTM(units, return_sequences=True)(RepeatVector(inputDim[0])(encoded))
					elif i > 1: 
						decoded = LSTM(units, return_sequences=True)(decoded)
			else: 
				raise ValueError("The codDim must be over 0.")

			decoded = LSTM(inputDim[-1], return_sequences=True)(decoded)
			self.model = Model(inputData, decoded)

		elif self.codeLayerType == 'dense': 
			assert len(inputDim) == 1
			inputData = Input(shape=(inputDim[0],))
			encoded = inputData
			for i, units in enumerate(codeDim): 
				encoded = Dense(units, activation='relu')(encoded)
			decoded = Dense(inputDim[-1], activation='sigmoid')(encoded)
			self.model = Model(inputData, decoded)

		elif self.codeLayerType == 'cov':
			pass


	def modelMasking(self, codeLayerType, inputDim, codeDim):

		self.codeLayerType = codeLayerType
		assert len(codeDim) > 0

		if self.codeLayerType == 'lstm':
			assert len(inputDim) == 2
			inputData = Input(shape=(inputDim[0],inputDim[1]))
			mask = Masking(mask_value=0.)(inputData)
			if len(codeDim) == 1:
				encoded = LSTM(codeDim[0])(mask)
				decoded = RepeatVector(inputDim[0])(encoded)
			elif len(codeDim) > 1:
				encoded = mask
				for i, units in enumerate(codeDim):
					if i == len(codeDim) - 1:
						 encoded = LSTM(units)(encoded)
						 continue		
					encoded = LSTM(units, return_sequences=True)(encoded)

				for i, units in enumerate(reversed(codeDim)): 
					if i == 1:
						decoded = LSTM(units, return_sequences=True)(RepeatVector(inputDim[0])(encoded))
					elif i > 1: 
						decoded = LSTM(units, return_sequences=True)(decoded)
			else: 
				raise ValueError("The codDim must be over 0.")

			decoded = LSTM(inputDim[-1], return_sequences=True)(decoded)
			self.model = Model(inputData, decoded)

		elif self.codeLayerType == 'cov': 
			pass
		elif self.codeLayerType == 'dense': 
			assert len(inputDim) == 1
			inputData = Input(shape=(inputDim[0],))
			# encoded = inputData
			# for i, units in enumerate(codeDim):
			# 	encoded = Dense(units, activation='relu')(encoded)
			# decoded = Dense(inputDim[-1], activation='sigmoid')(encoded)
			# self.model = Model(inputData, decoded)
			encoder = Dense(codeDim[0], activation="tanh",
							activity_regularizer=regularizers.l1(10e-5))(inputData)
			encoder = Dense(int(codeDim[0]/2), activation="relu")(encoder)
			decoder = Dense(int(codeDim[0]/2), activation='tanh')(encoder)
			decoder = Dense(inputDim[0], activation='relu')(decoder)
			self.model = Model(inputData, decoder)

	def compile(self, *args):

		if len(args) == 0:
			self.model.compile(optimizer=self.optimizer, loss=self.loss)
		elif len(args) == 1:
			if args[0] == 'temporal':
				self.sample_weight_mode = args[0]
				self.model.compile(optimizer=self.optimizer, loss=self.loss, sample_weight_mode=self.sample_weight_mode)
			elif args[0] == 'customFunction':
				self.model.compile(optimizer=self.optimizer, loss= self.weighted_vector_mse)
			else: 
				raise ValueError("Invalid maskType, please input 'sampleWeights' or 'customFunction'")
		else: 
			raise ValueError("argument # must be 0 or 1.")


	def fit(self, *args):

		# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
		if len(args) == 2:	
			if args[1] == 'nor':
				self.model.fit(args[0],
				args[0],	
				nb_epoch=self.nb_epoch, 
				batch_size=self.batch_size, 
				shuffle=self.shuffle, 
				validation_split=self.validation_split)
				# callbacks = [early_stopping])
			elif args[1] == 'rev':
				self.model.fit(args[0],
				np.flip(args[0], 1), 
				nb_epoch=self.nb_epoch, 
				batch_size=self.batch_size, 
				shuffle=self.shuffle, 
				validation_split=self.validation_split)
				# callbacks=[early_stopping])
			else: 
				raise ValueError("decoding sequence type: 'normal' or 'reverse'.")

		elif len(args) == 3:
			self.sampleWeights = args[2]	
			if args[1] == 'nor':
				self.model.fit(args[0],
				args[0],	
				nb_epoch=self.nb_epoch, 
				batch_size=self.batch_size, 
				shuffle=self.shuffle, 
				validation_split=self.validation_split, 
				sample_weight=self.sampleWeights)
				# callbacks=[early_stopping])
			elif args[1] == 'rev':
				self.model.fit(args[0],
				np.flip(args[0], 1), 
				nb_epoch=self.nb_epoch, 
				batch_size=self.batch_size, 
				shuffle=self.shuffle, 
				validation_split=self.validation_split,
				sample_weight=self.sampleWeights)
				# callbacks=[early_stopping])
			else: 
				raise ValueError("Please input, 'data', 'nor' or 'rev', 'sample_weights'")

	def predict(self, data):
		return self.model.predict(data)

	def weighted_vector_mse(self, y_true, y_pred):
		
		self.y_true = y_true
		self.y_pred = y_pred

		weight = T.ceil(self.y_true)
		loss = T.square(weight * (self.y_true - self.y_pred)) 
		# use appropriate relations for other objectives. E.g, for binary_crossentropy: 
		#loss = weights * (y_true * T.log(y_pred) + (1.0 - y_true) * T.log(1.0 - y_pred))
		return T.mean(T.sum(loss, axis=-1))




