import torch
import torch.nn.functional as F

def hotEncoding(dSet,device):
	labels = [doc[1] for doc in dSet]

	unique_labels = sorted(set(labels))
	label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
	print("Label mapping:", label_to_idx)

	label_indices = torch.tensor([label_to_idx[lbl] for lbl in labels],device=device)

	hotLabels = F.one_hot(label_indices,num_classes=len(unique_labels)).float().to(device)
	return hotLabels

		
def docsToMatrix(dSet,device):
	tensors = [doc[0] for doc in dSet]
	n = len(tensors)
	b = torch.ones(n,1)
	featureMatrix = torch.stack(tensors)
	featureMatrix = torch.cat((featureMatrix,b),dim=1).to(device)
	
	return featureMatrix

def getLogit(a, w):
	z = torch.mm(a,w)
	return z
	
def tanh(z):
	return torch.tanh(z)
def softmax(z):
	g = torch.nn.Softmax(dim=1)
	return g(z)

def getLoss(a,y):
	loss = F.cross_entropy(a,y)
	return(loss)

