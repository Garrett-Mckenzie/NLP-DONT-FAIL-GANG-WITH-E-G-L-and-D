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


def prep():

    gCorpus = None
    with open('../data/garrettCorpus.txt' , 'r') as file:
        gCorpus = file.read()
    gCorpus = gCorpus.split("\n")
   
    dCorpus = None
    with open('../data/delaneyCorpus.txt' , 'r') as file:
        dCorpus = file.read()
    dCorpus = dCorpus.split("\n")

    rCorpus = None
    with open('../data/riderCorpus.txt' , 'r') as file:
        rCorpus = file.read()
    rCorpus = rCorpus.split("\n")

    #ETHAN is a nerd for making it ?.!
    eCorpus = None
    with open('../data/ethanCorpus.txt' , 'r') as file:
        eCorpus = file.read()
    eCorpus = re.split(r"[!.?]+",eCorpus)

    corpi = [gCorpus,dCorpus,rCorpus,eCorpus]
    targetMap = {0:"g" , 1:"d" , 2:"r" , 3:"e"}

    #find min length of corpi
    min_len = None
    for arr in corpi:
        if arr == None:
            raise Exception("Something went wrong when reading files")
        if (min_len == None or len(arr) < min_len):
            min_len = len(arr)

    #shuffle the docs and put them all in dataframes
    i = 0
    df_corpi = []
    for corpus in corpi:
        random.shuffle(corpus)
        target = np.zeros(len(corpus)) + i
        data = {"docs":corpus , "target":target}
        df_corpi.append(pd.DataFrame.from_dict(data,orient = "columns"))
        i += 1
    
    #get IDF scores
    IDFScores = getIDF(df_corpi)

    #sample,stack,split, and shuffle
    data = pd.DataFrame()
    for df in df_corpi:
        sample = df.sample(min_len) 
        data = pd.concat([data,sample],ignore_index = True)
    trainX,testX,trainY,testY= train_test_split(data["docs"],data["target"] , train_size = 0.8)

    return ((trainX,trainY) , (testX,testY) , targetMap , IDFScores)

def encode():
    #get data,embeds, and tokenizer
    nlp = spacy.blank("en")
    embeds = KeyedVectors.load("glove_embeddings.data")
    train,test,labelMap,idf = prep()
    trainX , trainY = train
    testX , testY = test

    #encode training data
    totalTokens = 0
    inEmbed = 0
    total = len(trainX)
    encodedTrainX = []
    for i in tqdm(range(total)):
        doc = trainX.iloc[i]
        centroid = np.zeros(50)
        for token in nlp(doc):

            if not token.is_stop and not re.search(r"\s+",str(token).lower()): 
                vec , hasEmbed = getEmbed(embeds,str(token).lower())
                inEmbed += hasEmbed
                totalTokens += 1

                idfScore = 1
                if token in idf["token"]:
                    idfScore = idf[idf["token"] == token]["idf"]

                if hasEmbed != 0:
                    centroid += (vec/(LA.norm(vec,2.0))) * idfScore

        encodedTrainX.append(centroid)
    print(f"{(inEmbed/totalTokens)*100}% of tokens in training had an embedding in GLOVE")

    #encode testing data
    totalTokens = 0
    inEmbed = 0
    total = len(testX)
    encodedTestX = []
    for i in tqdm(range(total)):
        doc = testX.iloc[i]
        centroid = np.zeros(50)
        for token in nlp(doc):

            if not token.is_stop and not re.search(r"\s+",str(token).lower()): 
                vec,hasEmbed = getEmbed(embeds,str(token).lower())
                inEmbed += hasEmbed
                totalTokens += 1

                idfScore = 1
                if token in idf["token"]:
                    idfScore = idf[idf["token"] == token]["idf"]

                if hasEmbed != 0:
                    centroid += vec/(LA.norm(vec,2.0))

        encodedTestX.append(centroid)
    print(f"{(inEmbed/totalTokens)*100}% of tokens in testing had an embedding in GLOVE")

    encodedTrainX = np.array(encodedTrainX)
    trainY = np.array(trainY)

    encodedTestX = np.array(encodedTestX)
    testY = np.array(testY)

    np.savetxt("encodedTrainX.txt" , encodedTrainX ,delimiter = ',')
    np.savetxt("trainY.txt" , trainY ,delimiter = ',')
    np.savetxt("encodedTestX.txt" , encodedTestX ,delimiter = ',')
    np.savetxt("testY.txt" , testY ,delimiter = ',')

    print("!! All Done !!")

def getEmbed(embeds,token):
    vec = None
    hasEmbed = 0
    try:
        vec = embeds[token]
        hasEmbed += 1
    except Exception as e:
        vec = np.zeros(50)
    return vec,hasEmbed

def getIDF(corpi):
    data = pd.DataFrame()
    for corpus in corpi:
        data = pd.concat([data,corpus],ignore_index = True)

    vec = CountVectorizer(analyzer = 'word',stop_words = 'english')
    counts = vec.fit_transform(data["docs"])
    docFrequency = np.array((counts.sum(axis = 0)))[0]
    n = len(data)
    idf = np.log(n/docFrequency)  

    tokens = vec.get_feature_names_out()
    data = np.array([idf,tokens])
    data = np.transpose(data)

    returnMe = pd.DataFrame(data = data , columns = ["idf" , "token"])
    return returnMe
           
if __name__=="__main__":
    encode()   

