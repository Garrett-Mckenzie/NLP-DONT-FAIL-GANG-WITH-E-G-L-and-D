import re
import string

def createDocs(pathToFile,delimiters):
	#delimiters = [r"\.\.\.", r"\.", r"\?", r"!", r"\n", r"<EOD>"]
    with open(pathToFile,"r",encoding="utf-8") as f:
        text = f.read()
	#split to get documents by given delimiters
	print("fetching documents")
    pattern = r"(?:{})(?:\s+|$)".format("|".join(delimiters)) 
    sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]

	#clean and filter
	i = 0
	translator = str.maketrans('', '', string.punctuation)
	for sentence in sentences:
		#drop punc
		sentences[i] = text.translate(translator)
		#drop if empty
		if not sentence:
			del sentence[i]		

		if i%10 == 0:
			print("cleaning documents")
		i+=1
	return sentences

    
