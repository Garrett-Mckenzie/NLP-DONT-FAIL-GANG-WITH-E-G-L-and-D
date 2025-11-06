from train import *
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import json

if __name__=="__main__":
    model = Model()
    model.load("w1.pt","w2.pt","w3.pt","w4.pt")
    
    train,test = load_data()
    trainX,trainY = train
    testX,testY = test

    targetMap = None
    with open('targetMap.json' , 'r') as file:
        targetMap = json.load(file)

    predictions = model.predict(testX)
    print("\n")
   
    accuracy = accuracy_score(testY,predictions)
    print(f"The model's test accuracy was {accuracy}")

    precision = precision_score(testY,predictions,average="macro")
    print(f"The model's test precision was {precision}")

    recall = recall_score(testY,predictions,average="macro")
    print(f"The model's test recall was {recall}")

    f1 = f1_score(testY,predictions,average="macro")
    print(f"The model's test f1_macro score was {f1}")

    ps = precision_score(testY,predictions,average=None)
    rs = recall_score(testY,predictions,average=None)
    f1s = f1_score(testY,predictions,average=None)

    print("\n")
    for i in range(len(ps)):
        print(f"For the {targetMap[str(i)]} group we have the below scores")
        print(f"Precision: {ps[i]}")
        print(f"Recall: {rs[i]}")
        print(f"F1: {f1s[i]}")
        print("\n")
