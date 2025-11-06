from train import *

model = Model()
model.load("w1.pt","w2.pt","w3.pt","w4.pt")

while (True):
    print("\n")
    print("Welcome to interact.py by GM!")
    print("Simply type in your sentence and you will get probabilities of it belonging to each corpus.")
    print("Type quit to exit")
    x = input()
    if x == "quit":
        break
    else:
        model.predictProbInfrence(x)
