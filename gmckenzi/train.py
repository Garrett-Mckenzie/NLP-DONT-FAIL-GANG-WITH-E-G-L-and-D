from numpy import linalg as LA
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import *
import random
import torch
import tensorflow as tf
import torch.nn as nn
import json
from prep import getEmbed

class Model:

    #define weights and loss function
    def __init__(self):
        self.w1 = torch.rand(51,25, requires_grad=True,dtype=torch.float64)
        self.w2 = torch.rand(25,15, requires_grad=True,dtype=torch.float64)
        self.w3 = torch.rand(15,10, requires_grad=True,dtype=torch.float64)
        self.w4 = torch.rand(10,4, requires_grad=True,dtype=torch.float64)
        
        self.lossFunc = nn.CrossEntropyLoss()

        self.targetMap = None
        with open('targetMap.json' , 'r') as file:
            self.targetMap = json.load(file)
        self.idf = pd.read_csv("IDF.csv")

        self.nlp = spacy.blank("en")
        self.embeds = KeyedVectors.load("glove_embeddings.data")

    #make the first value 1 for bias
    def prep(self, X):
        X = np.insert(X , 0 ,np.ones(len(X)) , axis = 1)
        return torch.tensor(X,requires_grad=False,dtype=torch.float64)

    def forward(self, X , inTraining = 0):
        out1 = torch.tanh(torch.matmul(X,self.w1))
        out2 = torch.tanh(torch.matmul(out1,self.w2))
        out3 = torch.tanh(torch.matmul(out2,self.w3))
        out4 = torch.matmul(out3,self.w4)
        if inTraining:
            return out4
        else:
            return torch.softmax(out4,dim=1)

    def back(self, y_hat, y, lr , epoch):
        loss = self.lossFunc(y_hat , y)
        loss.backward()
        
        with torch.no_grad():
            self.w1 -= lr * self.w1.grad
            self.w1.grad = None

            self.w2 -= lr * self.w2.grad
            self.w2.grad = None

            self.w3 -= lr * self.w3.grad
            self.w3.grad = None

            self.w4 -= lr * self.w4.grad
            self.w4.grad = None

        print(f"loss was {loss} for the {epoch}th epoch")

    def train(self, X, y, lr , epochs): 
        x = self.prep(X)
        y = torch.tensor(y,requires_grad=False,dtype=torch.float64)
        for e in range(epochs):
            y_hat = self.forward(x,inTraining=1)
            self.back(y_hat,y,lr,e)

    def predict(self,X):
        x = self.prep(X)

        y_hat = None
        with torch.no_grad():
            y_hat = self.forward(x)
            y_hat = np.argmax(y_hat.detach().numpy() , axis=1)

        returnMe = np.zeros((len(x),4))
        for i in range(len(y_hat)):
            returnMe[i,y_hat[i]] += 1
        return returnMe
    
    def predictProbInfrence(self,X):
        X = self.prepInfrence(X)
        x = self.prep(X)

        y_hat = None
        with torch.no_grad():
            y_hat = self.forward(x)
            y_hat = y_hat.detach().numpy()

        y_hat = y_hat[0]
        print(y_hat)
        print(self.targetMap)
        for i,label in self.targetMap.items():
            print(f"Prob of {label} is {y_hat[int(i)] * 100}%")

        return y_hat

    def prepInfrence(self,txt):
        inEmbed = 0
        totalTokens = 0
        centroid = np.zeros(50)
        for token in self.nlp(txt):
            if not token.is_stop and not re.search(r"\s+" , str(token).lower()):
                vec , hasEmbed = getEmbed(self.embeds , str(token).lower())
                inEmbed += hasEmbed
                totalTokens += 1
                
                idfScore = 1
                if str(token).lower() in self.idf["token"]:
                    idfScore = idf[idf["token"] == str(token).lower()]
                
                if hasEmbed != 0:
                    centroid += vec/(LA.norm(vec,2.0)) * idfScore     
        centroid = centroid.reshape(1,50)
        print(f"{(inEmbed/totalTokens)*100}% of tokens in input had     an embedding in GLOVE")
        return centroid

         

    def save(self):    
        torch.save(self.w1 , "w1.pt")
        torch.save(self.w2 , "w2.pt")
        torch.save(self.w3 , "w3.pt")
        torch.save(self.w4 , "w4.pt") 

    def load(self,w1,w2,w3,w4):
        self.w1 = torch.load(w1)
        self.w2 = torch.load(w2)
        self.w3 = torch.load(w3)
        self.w4 = torch.load(w4)

def load_data():
    trainX = np.loadtxt("encodedTrainX.txt" , delimiter = ',') 
    trainY = np.loadtxt("trainY.txt" , delimiter = ',') 
    testX = np.loadtxt("encodedTestX.txt" , delimiter = ',') 
    testY = np.loadtxt("testY.txt" , delimiter = ',') 

    #prep work
    temp = np.zeros((len(trainY) , 4))
    for i in range(len(trainY)):
       temp[i,int(trainY[i])] += 1 
    trainY = np.copy(temp)
    
    #prep work
    temp = np.zeros((len(testY) , 4))
    for i in range(len(testY)):
       temp[i,int(testY[i])] += 1 
    testY = np.copy(temp)

    del temp

    return ((trainX , trainY) , (testX , testY))
 
if __name__=="__main__":
    train,test = load_data()
    trainX , trainY = train
    testX , testY = test

    model = Model()
    #lmao I love overfitting
    model.train(trainX,trainY,1.5,10000)   
    model.save()
