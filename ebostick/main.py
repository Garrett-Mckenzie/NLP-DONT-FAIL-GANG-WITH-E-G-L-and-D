import prep as p
import training as t
import numpy as np
import torch
import re
import sys
from gensim.models import KeyedVectors

def main():
	device = "cpu"
	prep = True
	train = True
	path1 = ""

	try:
		if sys.argv[1] == "-e":
			prep = False
			train = False
			print("skipping prep and training...")
			path1 = sys.argv[2]
			path2 = sys.argv[3]
			path3 = sys.argv[4]
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

		print("or skip prep with saved data:\npython main.py -s [pathToDataSplit] optional:[gpu|cpu]")
		print("or skip training with saved data:\npython main.py -e [pathToDataSplit] [pathToWeights1] [pathToWeights2] optional:[gpu|cpu]")

	if train:	
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

			print("or skip prep with saved data:\npython main.py -s [pathToDataSplit] optional:[gpu|cpu]")
			print("or skip training with saved data:\npython main.py -e [pathToDataSplit] [pathToWeights1] [pathToWeights2] optional:[gpu|cpu]")
		
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

			print("or skip prep with saved data:\npython main.py -s [pathToDataSplit] optional:[gpu|cpu]")
			print("or skip training with saved data:\npython main.py -e [pathToDataSplit] [pathToWeights1] [pathToWeights2] optional:[gpu|cpu]")

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


	if train:
		if device == "gpu" and torch.cuda.is_available():
			torch_device = torch.device("cuda")
			print("Using GPU acceleration.")
		else:
			torch_device = torch.device("cpu")
			torch.set_num_threads(torch.get_num_threads())	# enables CPU multithreading
			print(f"Using CPU with {torch.get_num_threads()} threads.")
		print("encoding labels...")
		labels = t.hotEncoding(splits[0],torch_device)
		print("creating feature matrix...")
		xTrain = t.docsToMatrix(splits[0],torch_device)
		w1 = torch.rand(xTrain.size()[1],24,requires_grad=True,device=torch_device)
		w2 = torch.rand(24,len(labels[0]),requires_grad=True,device=torch_device)

		print("training...")
		eta = .03
		lossDeltaThresh = 0.00001
		max_iters = 10000
		cur_iter=0

		prevLoss = 100	
		for cur_iter in range(max_iters):
			# Forward pass
			z1 = t.getLogit(xTrain, w1)
			a1 = t.tanh(z1)
			z2 = t.getLogit(a1, w2)
			loss = t.getLoss(z2, labels)

			# Backpropagation
			loss.backward()

			# Weight update
			with torch.no_grad():
				w1 -= eta * w1.grad
				w2 -= eta * w2.grad

				w1.grad = None
				w2.grad = None

			loss_value = loss.item()
			if abs(prevLoss - loss_value) < lossDeltaThresh:
				print(f"Converged at iter {cur_iter}, loss={loss_value:.4f}")
				break

			prevLoss = loss_value

			if cur_iter % 100 == 0:
				print(f"Iter {cur_iter}, Loss = {loss.item():.4f}")
		
		save = input("save weights(yes/no)?: ")
		if save.lower().strip() == "yes":
			w1 = w1.to('cpu')
			w2 = w2.to('cpu')
			torch.save(w1, "w1.pt")
			torch.save(w2, "w2.pt")
			print("--saved weights 1: w1.pt--")
			print("--saved weights 2: w2.pt--")
		else:
			print("<<skipping save>>")
		print("evaluating classifier with runtime weights...")

	else:
		print("--loading weights: w1.pt--")
		w1 = torch.load(path2)
		print("--loading weights: w2.pt--")
		w2 = torch.load(path3)
		print("evaluating classifier with saved weights...")
	
	### evaluate ###


if __name__ == "__main__":
	import multiprocessing
	multiprocessing.freeze_support()  # optional, but safe
	main()





