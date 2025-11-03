import prep as p
import sys
from gensim.models import KeyedVectors

useGpu = False

try:
    corporaPathes = sys.argv[1:5]
    try:
        g = sys.argv[5]
        useGpu = True
        print("utilizing GPU")
    except(Exception):
        print("utilizing strictly CPUs")

except(Exception):
    print("see run use:\npython main.py [path_to_corp1] [path_to_corp2] [path_to_corp3] [path_to_corp4] optional:[gpu|cpu]")

#embeddings
embeds = KeyedVectors.load("glove_embeddings.data")
