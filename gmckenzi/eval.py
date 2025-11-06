from train import *
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

if __name__=="__main__":
    model = Model()
    model.load("w1.pt","w2.pt","w3.pt","w4.pt")
    
    train,test = load_data()
    testX,testY = test

    predictions = model.predict(testX)

    accuracy = accuracy_score(testY,predictions)
    print(f"The model's accuracy was {accuracy}")

    precision = precision_score(testY,predictions,average="macro")
    print(f"The model's precision was {precision}")

    recall = recall_score(testY,predictions,average="macro")
    print(f"The model's recall was {recall}")

    f1 = f1_score(testY,predictions,average="macro")
    print(f"The model's f1_micro score was {f1}")
   
