import prep as p
import evaluate as e
import training as t
import numpy as np
import torch
import re
import sys
from gensim.models import KeyedVectors

def unsupervised(split,w1,w2,device):
	### evaluate ###
	if device == "gpu" and torch.cuda.is_available():
		torch_device = torch.device("cuda")
		print("Using GPU acceleration.")
	else:
		torch_device = torch.device("cpu")
		torch.set_num_threads(torch.get_num_threads())	# enables CPU multithreading
		print(f"Using CPU with {torch.get_num_threads()} threads.")

	print("encoding labels...")
	labels = t.hotEncoding(split,torch_device)
	print("creating feature matrix...")
	xTest = t.docsToMatrix(split,torch_device)
	
	# Forward pass
	z1 = t.getLogit(xTest, w1)
	a1 = t.tanh(z1)
	z2 = t.getLogit(a1, w2)
	a2 = t.softmax(z2)

	yHat = torch.argmax(a2,dim=1)
	y = labels.argmax(dim=1)
	n = len(yHat)
	correct = (y == yHat).sum().item()
	recalls =[]
	precisions = []
	f1s = []

	i =0
	print("see label mappings above")
	while i< len(labels[0]):
		tp,fp,fn = e.tpFpFn(i,yHat,y)
		recall = round((tp/(tp+fn))*100,2)
		recalls.append(recall)
		precision = round((tp/(tp+fp))*100,2)
		precisions.append(precision)
		f1 = round((2*precision*recall)/(precision+recall),2)
		f1s.append(f1)

		print(f"Corpus {i}")
		print(f"	recall = {recall}%")
		print(f"	precision = {precision}%")
		print(f"	f1-score = {f1}%")
		i+=1

	print(f"overall:")
	print(f"	accuracy = {round((correct/n)*100,2)}%")
	print(f"	recall avg = {round(sum(recalls)/len(labels[0]),2)}%")
	print(f"	precision avg = {round(sum(precisions)/len(labels[0]),2)}%")
	print(f"	f1 avg = {round(sum(f1s)/len(labels[0]),2)}%")
	return

device = "cpu"
try:
	path1 = sys.argv[1]
	path2 = sys.argv[2]
	path3 = sys.argv[3]
	path4 = sys.argv[4]
	print("--loading data: idfs.pt--")
	idfs = torch.load(path1)
	print("--loading data: dataSplits.pt--")
	splits = torch.load(path2)
	print("--loading weights: w1.pt--")
	w1 = torch.load(path3)
	print("--loading weights: w2.pt--")
	w2 = torch.load(path4)

	try:
		g = sys.argv[4]
		if(g.lower()=="cpu"):
			raise ExceptionType("using cpu")
		device = "gpu"
	except(Exception):
		device = "cpu"
except(Exception):
	print("see use:\npython interact.py [pathToIdfs] [pathToDataSplit] [pathToWeights1] [pathToWeights2] optional:[gpu|cpu]")
	sys.exit(0)


try:
	embeds = KeyedVectors.load("glove_embeddings.data")
except(Exception):
	print("no saved embeddings: get the 100-glove embedds")
	sys.exit(0)

print("enter a string to get the 4-simplex probabilities for that string (or enter '.s' to see the results of the unsupervised split)")
print("'.x' to quit")
done = False

if device == "gpu" and torch.cuda.is_available():
	torch_device = torch.device("cuda")
else:
	torch_device = torch.device("cpu")
	torch.set_num_threads(torch.get_num_threads())	# enables CPU multithreading

w1 = w1.to(torch_device)
w2 = w2.to(torch_device)
splits[0] = [(centroid.to(torch_device), str(label)) for centroid, label in splits[0]]
splits[1] = [(centroid.to(torch_device), str(label)) for centroid, label in splits[1]]
splits[2] = [(centroid.to(torch_device), str(label)) for centroid, label in splits[2]]

while(not done):

	cmd = input("$: ")
	if cmd == '.x':
		break
	if cmd == '.s':
		unsupervised(splits[2],w1,w2,device)
	else:
#displays label mapping based on training
		labels = t.hotEncoding(splits[0],torch_device)
		#encode cmd
		centroid = p.docCentroid([cmd],embeds,idfs,torch_device)
		xIn = t.docsToMatrix([[centroid]],torch_device)

		z1 = t.getLogit(xIn, w1)
		a1 = t.tanh(z1)
		z2 = t.getLogit(a1, w2)
		a2 = t.softmax(z2)
		print(a2[0])
		
		















