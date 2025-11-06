
You need the size 100 embeddings from gensim

Files:
	main.py
	interact.py
	prep.py
	evaluate.py
	training.py
To Run:
`python main.py [path_to_ethanCorp] [path_to_garrettCorp] [path_to_delaneyCorp] [path_to_laurCorp] optional:[gpu|cpu]`

or skip prep with saved data: `python main.py -s [pathToDataSplit] optional:[gpu|cpu]`
or skip training with saved data x2:`python main.py -e [pathToDataSplit] [pathToWeights1] [pathToWeights2] optional:[gpu|cpu]")`

**EX**:python main.py ../data/ethanCorpus.txt ../data/garrettCorpus.txt ../data/delaneyCorpus.txt ../data/riderCorpus.txt gpu

To Run Interact.py:
`python interact.py [pathToIdfs] [pathToDataSplit] [pathToWeights1] [pathToWeights2] optional:[gpu|cpu]`

**EX**: python interact.py idfs.pt dataSplits.pt w1.pt w2.pt cpu

**NOTE**: GPU option expects nvidia cuda
