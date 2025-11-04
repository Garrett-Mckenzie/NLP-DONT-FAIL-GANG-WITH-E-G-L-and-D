# prep.py
import re
import string
import os
from concurrent.futures import ProcessPoolExecutor
import torch


def clean_sentence(sentence, translator):
	"""Remove punctuation and whitespace from one sentence (CPU)."""
	return sentence.translate(translator).strip()

def clean_sentence_wrapper(args):
	s, translator = args
	return clean_sentence(s, translator)


def createDocs(pathToFile, delimiters, device="cpu", workers=None):
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
	pattern = r"(?:{})(?:\s+|$)".format("|".join(delimiters))
	sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]

	translator = str.maketrans('', '', string.punctuation)

	if device == "cpu":
		# Default to using all but one CPU core
		if workers is None:
			workers = max(1, os.cpu_count() - 1)

		print(f"Cleaning {len(sentences)} documents using {workers} CPU workers...")
		with ProcessPoolExecutor(max_workers=workers) as executor:
			cleaned = list(executor.map(clean_sentence_wrapper, [(s, translator) for s in sentences]))


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
		cleaned = [s.strip() for s in cleaned_text.split("\n") if s.strip()]

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

