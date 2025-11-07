import torch

#calc false positives
#yHat indices of guess
#y one hot enoding of correct answers
def tpFpFn(theClass, yHat,y_true):
	tp = ((yHat == theClass) & (y_true == theClass)).sum().item()
	fp = ((yHat == theClass) & (y_true != theClass)).sum().item()
	fn = ((yHat != theClass) & (y_true == theClass)).sum().item()
	return tp, fp, fn


