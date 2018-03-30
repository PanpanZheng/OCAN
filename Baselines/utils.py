'''
    Author: Panpan Zheng
    Date created:  1/15/2018
    Python Version: 2.7
'''

import numpy as np
from numpy import random
import pandas as pd
from collections import defaultdict
import datetime
from datetime import * 
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



# functions for data processing. 

def getUserDict(f_users):

	users_frame = pd.read_csv(f_users, sep='\t', header=None)
	users_frame = users_frame.applymap(str)
	user2id = users_frame.set_index(0)[1].to_dict()

	uid2type = users_frame.set_index(1)[4].to_dict()

	for uid in uid2type:
		if uid2type[uid] == "benign":
			uid2type[uid] = 0
		else:
			uid2type[uid] = 1

	id2user = users_frame.set_index(1)[0].to_dict()
	return user2id, uid2type, id2user


def getPageDict(f_pages):

	pages_frame = pd.read_csv(f_pages, sep=',')
	pages_frame = pages_frame.applymap(str)
	page2id = pages_frame.set_index('pagetitle')['pageid'].to_dict()
	id2Cgr = pages_frame.set_index('pageid')['pagecategories'].to_dict()
	id2page = pages_frame.set_index('pageid')['pagetitle'].to_dict()
	return page2id, id2Cgr, id2page


def TleRevTim(files,user2id,page2id):
	titleSet = defaultdict(list)
	revSet = defaultdict(list)
	timSet = defaultdict(list)
	for f in files:
		df = pd.read_csv(f,sep=',')
		for index, row in df.iterrows():
			try:
				usrid = user2id[row['username']]
			except Exception as e:
				print row['username']
			try:
				pageid = page2id[row['pagetitle']]
			except Exception as e:
				print row['pagetitle']
			isRev = row['isReverted']
			revTime = row['revtime']
			titleSet[usrid].append(pageid)
			if isRev:			
				revSet[usrid].append(1)
			else:
				revSet[usrid].append(0)
			timSet[usrid].append(revTime)
	return titleSet, revSet, timSet



def getLabel(filesB,filesV,user2id,posLabel,negLabel):
	usrListB = list()
	usrListV = list()
	userid2Label = dict()
	for f in filesB:
		df = pd.read_csv(f,sep=',')
		for index, row in df.iterrows():
			usrid = user2id[row['username']]
			usrListB.append(usrid)
	usrListB = list(set(usrListB))
	for val in usrListB:
		userid2Label[val] = posLabel
	for f in filesV:
		df = pd.read_csv(f,sep=',')
		for index, row in df.iterrows():
			usrid = user2id[row['username']]
			usrListV.append(usrid)
	usrListV = list(set(usrListV))
	for val in usrListV:
		userid2Label[val] = negLabel
	return userid2Label


def train_test_split(*agrs):

	trainRatio = agrs[-1]
	assert trainRatio <= 1.0
	size = agrs[0].shape[0]
	thre = int(trainRatio * 10)
	indSet = np.random.randint(0,10, size) + 1
	trainInd = [indSet <= thre]
	testInd = [indSet > thre]
	X_usrs, y, X_pages, X_tim, X_rev, X_len = \
									agrs[0], agrs[1], agrs[2], agrs[3], agrs[4], agrs[5]
	return X_usrs[trainInd], y[trainInd], X_pages[trainInd], \
									X_tim[trainInd], X_rev[trainInd], X_len[trainInd],\
		   X_usrs[testInd], y[testInd], X_pages[testInd], \
									X_tim[testInd], X_rev[testInd], X_len[testInd]


def MetaPageList(files,page2id):
	MPL = list()
	for f in files:
		df = pd.read_csv(f,sep=',')
		for index, row in df.iterrows():
			try:
				pageid = page2id[row['pagetitle']]
			except Exception as e:
				print row['pagetitle']
			title = row['pagetitle'].lower()
			if "user:" in title or "talk:" in title or "user talk:" in title or  "wikipedia" in title:
				MPL.append(pageid)
	return np.array(list(set(MPL)))

def encode(x, n):
	x = int(x)
	result = np.zeros(n).tolist()
	result[x] = 1.
	return result

def TimeDiff(p1, p2, timDiff):
	p1 = datetime.strptime(p1, '%Y-%m-%dT%XZ')
	p2 = datetime.strptime(p2, '%Y-%m-%dT%XZ')
	td = p2 - p1
	if td.days*24*60 + td.seconds/60 < timDiff:
		return 1
	else: 
		return 0


def IsMetaPage(p, metaDict):
	if p in metaDict: 
		return 1
	else: 
		return 0



# functions for model training and data analysis

def recMSE(a,b):
	return np.mean((a-b)**2)

def recErr(X1, X2):
	seqRecErr = list()
	for s1, s2 in zip(X1, X2):
		seqRecErr.append(np.sum((s1 - s2)**2)/float(np.prod(s1.shape)))
		# seqRecErr.append(np.mean([LA.norm(x1 - x2)**2/8. for x1, x2 in zip(s1, s2)]))
		# seqRecErr.append(np.mean([LA.norm(x1 - x2) for x1, x2 in zip(s1, s2)]))
	return np.array(seqRecErr)

def recErrMeaVar(X):
	return np.mean(X), np.var(X)

def recErrHist(*args):

	plt.figure()
	# # axes = plt.gca()
	# # axes.set_xlim([0.,1.0])
	# # axes.set_ylim([ymin,ymax])
	# plt.subplot(3,1,1)
	# plt.title('Trainning')
	# weights = np.ones_like(args[0])/len(args[0])
	# plt.hist(args[0], weights=weights, bins=100)
	# plt.ylabel('Probability')
	# axes = plt.gca()
	# axes.set_xlim([0.,.1])

	plt.subplot(2,1,1)
	plt.title('Benign')
	weights = np.ones_like(args[0])/len(args[0])
	plt.hist(args[0], weights=weights, bins=100)
	plt.ylabel('Probability')
	axes = plt.gca()
	axes.set_xlim([0.,.1])
	
	plt.subplot(2,1,2)
	plt.title('Vandal')
	weights = np.ones_like(args[1])/len(args[1])
	plt.hist(args[1], weights=weights, bins=100)
	plt.ylabel('Probability')
	axes = plt.gca()
	axes.set_xlim([0.,.1])

	plt.xlabel('Recontruction Error')
	plt.show()


def vanDet(X,thrd):
	return np.sum(X>=thrd)/float(len(X))

def vanDet2(X,thrd):
	return (np.array(X)>=thrd).astype(int)

def TSNE_2D_show_bi(X,y,i):
	model = TSNE(n_components=2, random_state=0)
	X_2D = model.fit_transform(X)
	X_2D_beg = X_2D[y == 1]
	X_2D_val = X_2D[y == 0]

	fig = plt.figure()
	fig.patch.set_facecolor('w')
	ax = fig.add_subplot(111)
	ax.set_axis_off()

	blue_dot, = plt.plot(X_2D_beg[:,0],X_2D_beg[:,1], "ro", mec='none')
	red_dot, = plt.plot(X_2D_val[:,0],X_2D_val[:,1], "bo", mec='none')
	plt.legend([blue_dot, red_dot], ["Benign", "Vandal"], numpoints=1)
	plt.savefig("representation_%s"%i)
	plt.clf()
	plt.close()

def TSNE_2D_show_tri(X,y):
	model = TSNE(n_components=2, random_state=0)
	X_2D = model.fit_transform(X)
	X_2D_train_beg = X_2D[y == 1]
	X_2D_fake = X_2D[y == 0]
	X_2D_val = X_2D[y == 2]


	fig = plt.figure()
	fig.patch.set_facecolor('w')
	ax = fig.add_subplot(111)
	ax.set_axis_off()

	red_dot, = plt.plot(X_2D_train_beg[:,0],X_2D_train_beg[:,1], "ro", mec='none')
	green_dot, = plt.plot(X_2D_fake[:, 0], X_2D_fake[:, 1], "go", mec='none')
	blue_dot, = plt.plot(X_2D_val[:, 0], X_2D_val[:, 1], "bo", mec='none')
	# yellow_dot, = plt.plot(X_2D_train[:, 0], X_2D_train[:, 1], "yo", mec='none')
	plt.legend([red_dot, green_dot, blue_dot], ["Benign", "Fake", "Vandal"], numpoints=1)
	plt.show()
	plt.clf()
	plt.close()
	return X_2D_train_beg, X_2D_fake, X_2D_val




def draw_trend(D_real_prob, D_fake_prob, D_val_prob, fm_loss, f1):

    fig = plt.figure()
    fig.patch.set_facecolor('w')
    # plt.subplot(311)
    p1, = plt.plot(D_real_prob, "-g")
    p2, = plt.plot(D_fake_prob, "--r")
    p3, = plt.plot(D_val_prob, ":c")
    plt.xlabel("# of epoch")
    plt.ylabel("probability")
    leg = plt.legend([p1, p2, p3], [r'$p(y|V_B)$', r'$p(y|\~{V})$', r'$p(y|V_M)$'], loc=1, bbox_to_anchor=(1, 1), borderaxespad=0.)
    leg.draw_frame(False)
    # plt.legend(frameon=False)

    fig = plt.figure()
    fig.patch.set_facecolor('w')
    # plt.subplot(312)
    p4, = plt.plot(fm_loss, "-b")
    plt.xlabel("# of epoch")
    plt.ylabel("feature matching loss")
    # plt.legend([p4], ["d_real_prob", "d_fake_prob", "d_val_prob"], loc=1, bbox_to_anchor=(1, 1), borderaxespad=0.)

    fig = plt.figure()
    fig.patch.set_facecolor('w')
    # plt.subplot(313)
    p5, = plt.plot(f1, "-y")
    plt.xlabel("# of epoch")
    plt.ylabel("F1")
    # plt.legend([p1, p2, p3, p4, p5], ["d_real_prob", "d_fake_prob", "d_val_prob", "fm_loss","f1"], loc=1, bbox_to_anchor=(1, 3.5), borderaxespad=0.)
    plt.show()


def sample_shuffle(X):
	# n_samples = X.shape[0]
	n_samples = len(X)
	s = np.arange(n_samples)
	np.random.shuffle(s)
	return np.array(X[s])

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
