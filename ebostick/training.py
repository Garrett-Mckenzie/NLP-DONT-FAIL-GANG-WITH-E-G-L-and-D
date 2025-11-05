import torch
import torch.nn.functional as F

def hotEncoding(dSet):
	labels = [doc[1] for doc in dSet]

	unique_labels = sorted(set(labels))
	label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
	print("Label mapping:", label_to_idx)

	label_indices = torch.tensor([label_to_idx[lbl] for lbl in labels])

	hotLabels = F.one_hot(label_indices,num_classes=len(unique_labels)).float()
	return hotLabels
		
def docsToMatrix(dSet):
	tensors = [doc[0] for doc in dSet]
	n = len(tensors)
	b = torch.ones(n,1)
	featureMatrix = torch.stack(tensors)
	featureMatrix = torch.cat((featureMatrix,b),dim=1)
	
	return featureMatrix

def getLogit(a, w,device):
	z = torch.mm(a,w)
	return z
	
def tanh(z,device):
	return (torch.exp(z)-torch.exp(-1*z))/(torch.exp(z)+torch.exp(-1*z))
def softmax(z, device):
	g = torch.nn.Softmax(dim=1)
	return g(z)

def getLoss(a,y):
	loss = F.cross_entropy(a,y)
	return(loss)

