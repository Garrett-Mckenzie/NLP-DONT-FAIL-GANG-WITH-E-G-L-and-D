# prep.py
import numpy as np
import re
import string
import os
from concurrent.futures import ProcessPoolExecutor
import torch
from torch.utils.data import random_split


def clean_sentence(sentence, translator):
	"""Remove punctuation and whitespace from one sentence (CPU)."""
	return sentence.translate(translator).strip()

def clean_sentence_wrapper(args):
	s, translator,corpName = args
	return (clean_sentence(s, translator),corpName)


def createDocs(pathToFile, delimiters,corpName, device="cpu", workers=None):
	"""
	Reads a text file, splits it into sentences by delimiters,
	and cleans punctuation/whitespace.

	Args:
		pathToFile: path to text file
		delimiters: list of regex patterns to split text
		device: "cpu" (default) or "gpu" to run cleaning
		workers: number of CPU processes (optional)
	"""
	with open(pathToFile, "r", encoding="utf-8") as f:
		text = f.read()

	print(f"Fetching documents using {device.upper()}...")

	# Split text into sentences
	pattern = r"(?:{})".format("|".join(delimiters))
	sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]
	translator = str.maketrans('', '', string.punctuation)

	if device == "cpu":
		# Default to using all but one CPU core
		if workers is None:
			workers = max(1, os.cpu_count() - 1)

		print(f"Cleaning {len(sentences)} documents using {workers} CPU workers...")
		with ProcessPoolExecutor(max_workers=workers) as executor:
			cleaned = list(executor.map(
				clean_sentence_wrapper,
				((s, translator, corpName) for s in sentences)
			))


		print("Done (CPU).")
		return cleaned

	elif device == "gpu":
		if not torch.cuda.is_available():
			raise RuntimeError("GPU requested but no CUDA device found.")

		print(f"Cleaning {len(sentences)} documents on GPU...")

		# Join sentences into one big tensor of chars for batch processing
		joined = "\n".join(sentences)
		data = torch.tensor(list(joined.encode("utf-8")), dtype=torch.uint8, device="cuda")

		# Remove punctuation (vectorized GPU op)
		punct_bytes = set(bytes(string.punctuation, "utf-8"))
		mask = torch.tensor([b not in punct_bytes for b in data.tolist()],
							dtype=torch.bool, device="cuda")
		filtered = data[mask]

		# Decode back to string and split again
		cleaned_text = filtered.cpu().numpy().tobytes().decode("utf-8")
		cleaned = [(s.strip(), corpName) for s in cleaned_text.split("\n") if s.strip()]

		print("Done (GPU).")
		return cleaned

	else:
		raise ValueError("device must be 'cpu' or 'gpu'.")


if __name__ == "__main__":
	# Example usage
	delimiters = [r"\.\.\.", r"\.", r"\?", r"!", r"\n", r"<EOD>"]

	# Try CPU first
	docs_cpu = createDocs("clnCorpus.txt", delimiters, device="cpu")
	print(f"CPU cleaned {len(docs_cpu)} documents.")

	# Then GPU (if available)
	if torch.cuda.is_available():
		docs_gpu = createDocs("clnCorpus.txt", delimiters, device="gpu")
		print(f"GPU cleaned {len(docs_gpu)} documents.")

def procrustes(corpora):
	lengths = [len(corpus) for corpus in corpora]
	smallest = min(lengths)

	newCorpora = []
	for corpus in corpora:
		corpus = np.array(corpus)  # shape (N, 2)
		idx = np.random.choice(len(corpus), size=smallest, replace=False)
		subset = corpus[idx]
		newCorpora.append(subset)
	return newCorpora

def sampleSplit(corpora, split):
	aggCorp = np.concatenate(corpora)
	n = len(aggCorp)

	# Compute lengths for each split
	lengths = [int(x * n) for x in split]
	lengths[0] += n - sum(lengths)	# Fix rounding errors

	# Shuffle indices once for randomness
	indices = np.random.permutation(n)

	# Create non-overlapping splits
	splits = []
	start = 0
	for length in lengths:
		end = start + length
		splits.append(aggCorp[indices[start:end]])
		start = end

	return splits

def buildVocab(training):
	vocab = set()
	for doc in training:
		words = doc[0].split()
		vocab.update(words)
	return vocab


def computeIdfs(training):
	N = len(training)
	vocab = buildVocab(training)
	idfs = dict.fromkeys(vocab, 0)

	for doc in training:
		sentence = doc[0]
		words_in_doc = set(sentence.split())
		for word in words_in_doc:
			if word in idfs:
				idfs[word] += 1

	for word, df in idfs.items():
		idfs[word] = (N / df) if df > 0 else 0.0	
	return idfs

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

def docCentroid(doc, embeddings, idfs, device):
    """
    Compute TF-IDF weighted centroid vector for a single document on the specified device.
    """
    vecSum = torch.zeros(100, device=device)
    tokens = doc[0].split()

    for word in tokens:
        if word in embeddings:
            v = torch.tensor(embeddings[word], dtype=torch.float32, device=device)
            idf = idfs.get(word, 1.0)
            vecSum += idf * (v / v.norm())

    if vecSum.norm() > 0:
        vecSum = vecSum / vecSum.norm()
    return vecSum


def encode(docs, idfs, embeddings, device='cpu', num_threads=8):
    """
    Encode a list of documents into (centroid_vector, label) pairs.
    Supports GPU ('gpu') or CPU ('cpu') devices.
    """
    # Map 'gpu' to 'cuda' for PyTorch
    if device.lower() == 'gpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    device = torch.device(device)

    newDocs = []

    if device.type == 'cuda':
        # GPU: single-threaded is usually fine, batching could be added later
        for doc in docs:
            centroid = docCentroid(doc, embeddings, idfs, device)
            newDocs.append((centroid,str(doc[1])))
    else:
        # CPU: multithreaded
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(docCentroid, doc, embeddings, idfs, device): doc for doc in docs}
            for future in as_completed(futures):
                doc = futures[future]
                centroid = future.result()
                newDocs.append((centroid, doc[1]))

    return newDocs




