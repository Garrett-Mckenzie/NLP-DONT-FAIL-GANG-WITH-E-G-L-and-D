import prep as p
import training as t
import numpy as np
import torch
import re
import sys
from gensim.models import KeyedVectors

device = "cpu"
prep = True
path1 = ""

try:
	if sys.argv[1] == "-s":
		prep = False
		print("skipping prep...")
		path1 = sys.argv[2]
		try:
			g = sys.argv[3]
			if(g.lower()=="cpu"):
				raise ExceptionType("using cpu")
			device = "gpu"
			print("utilizing GPU")
		except(Exception):
			print("utilizing strictly CPUs")
except(Exception):
	print("see run use:\npython main.py [path_to_ethanCorp] [path_to_garrettCorp] [path_to_delaneyCorp] [path_to_laurCorp] optional:[gpu|cpu]")

	print("or skip training with saved data:\npython main.py -s [pathToDataSplit] optional:[gpu|cpu]")
	
if prep:
	try:
		ePath = sys.argv[1]
		gPath = sys.argv[2]
		dPath = sys.argv[3]
		lPath = sys.argv[4]
		try:
			g = sys.argv[5]
			if(g.lower()=="cpu"):
				raise ExceptionType("using cpu")
			device = "gpu"
			print("utilizing GPU")
		except(Exception):
			print("utilizing strictly CPUs")

	except(Exception):
		print("see run use:\npython main.py [path_to_ethanCorp] [path_to_garrettCorp] [path_to_delaneyCorp] [path_to_laurCorp] optional:[gpu|cpu]")

		print("or skip training with saved data:\npython main.py -s [pathToDataSplit] optional:[gpu|cpu]")

#get embeddings
try:
	embeds = KeyedVectors.load("glove_embeddings.data")
except(Exception):
	print("no saved embeddings")

if prep:
#attempt prep corpora
	ethanDelim = [r"\.", r"\?", r"!"]
#garrDelim = [r"<EOD>"]
#gar now using delaney delimiters
	delaneyDelim = [r"\n"]
#laur same as delaney

	corpora = []
	corpora.append(np.array(p.createDocs(ePath,ethanDelim,"gdc",device)))
	corpora.append(np.array(p.createDocs(gPath,delaneyDelim,"finance",device)))		
	corpora.append(np.array(p.createDocs(dPath,delaneyDelim,"delaney",device)))		
	corpora.append(np.array(p.createDocs(lPath,delaneyDelim,"laur",device)))		

	print("chopping up corpora...")
	corpora = p.procrustes(corpora)
#[training, testing, unsupervised]
	print("splitting corpus into training, test, and unsupervised sets...")
	splits = p.sampleSplit(corpora,[0.7,.2,.1])
	print("computing IDFs for training set...")
	idfs = p.computeIdfs(splits[0])

#need to create encodings
	print("encoding sets...")
	splits[0] = p.encode(splits[0],idfs,embeds,device)
	splits[1] = p.encode(splits[1],idfs,embeds,device)
	splits[2] = p.encode(splits[2],idfs,embeds,device)

# Ensure all centroids are on CPU
	splits[0] = [(centroid.to('cpu'), str(label)) for centroid, label in splits[0]]
	splits[1] = [(centroid.to('cpu'), str(label)) for centroid, label in splits[1]]
	splits[2] = [(centroid.to('cpu'), str(label)) for centroid, label in splits[2]]

	print("--saved data: dataSplits.pt--")
	torch.save(splits, "dataSplits.pt")
else:
	print("--loading data: dataSplits.pt--")
	splits = torch.load(path1)

print("encoding labels...")
labels = t.hotEncoding(splits[0])
print("creating feature matrix...")
xTrain = t.docsToMatrix(splits[0])
w1 = torch.rand(xTrain.size()[1],4)

print("training...")
eta = .0001
lossDeltaThresh = 0.001
max_iters = 5000

z = t.getLogit(xTrain,w1,device)
xTrain = t.tanh(z,device)

w2 = torch.rand(xTrain.size()[1],4)

z = t.getLogit(xTrain,w2,device)
loss = t.getLoss(z, labels)
print(loss)
xTrain = t.softmax(z,device)






