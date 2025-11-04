import prep as p
import re
import sys
from gensim.models import KeyedVectors

useGpu = False

try:
	ePath = sys.argv[1]
	gPath = sys.argv[2]
	dPath = sys.argv[3]
	lPath = sys.argv[4]
	try:
		g = sys.argv[5]
		useGpu = True
		print("utilizing GPU")
	except(Exception):
		print("utilizing strictly CPUs")

except(Exception):
	print("see run use:\npython main.py [path_to_ethanCorp] [path_to_garrettCorp] [path_to_delaneyCorp] [path_to_laraCorp] optional:[gpu|cpu]")

#get embeddings
try:
	embeds = KeyedVectors.load("glove_embeddings.data")
except(Exception):
	print("no saved embeddings")

#attempt prep corpora
ethanDelim = [r"\.", r"\?", r"!"]
garrDelim = [r"<EOD>"]
delaneyDelim = [r"\n"]

corpora = {}
corpora[0] = p.createDocs(ePath,ethanDelim) 	
	
