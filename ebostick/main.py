import prep as p
import numpy as np
import re
import sys
from gensim.models import KeyedVectors

device = "cpu"

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

#get embeddings
try:
	embeds = KeyedVectors.load("glove_embeddings.data")
except(Exception):
	print("no saved embeddings")

#attempt prep corpora
ethanDelim = [r"\.", r"\?", r"!"]
#garrDelim = [r"<EOD>"]
#gar now using delaney delimiters
delaneyDelim = [r"\n"]
#laur same as delaney

corpora = []
corpora.append(np.array(p.createDocs(ePath,ethanDelim,device)))
corpora.append(np.array(p.createDocs(gPath,delaneyDelim,device)))		
corpora.append(np.array(p.createDocs(dPath,delaneyDelim,device)))		
corpora.append(np.array(p.createDocs(lPath,delaneyDelim,device)))		

print("chopping up corpora...")
corpora = p.procrustes(corpora)









