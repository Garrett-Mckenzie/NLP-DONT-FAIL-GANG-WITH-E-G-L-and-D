import re

def createDocs(pathToFile,delimiters):
#delimiters = [r"\.\.\.", r"\.", r"\?", r"!", r"\n", r"<EOD>"]
    with open(pathToFile,"r",encoding="utf-8") as f:
        text = f.read()
    pattern = r"(?:{})(?:\s+|$)".format("|".join(delimiters)) 
    sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]

    
