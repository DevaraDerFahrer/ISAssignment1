import torch
import time
import pickle
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Grayscale
from torchvision.transforms import Normalize
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as tNN
import torch.optim as tOptim
from sklearn.metrics import accuracy_score
import mylibrary
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
    
class ModelCollection:
    def __init__(self, device, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10):
        self.device = device
        self.numOfModels = 2
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        self.model8 = model8
        self.model9 = model9
        self.model10 = model10
    def Vote(self, x, batchSize):
        preds = []
        preds.append(torch.argmax(torch.softmax(self.model1(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model2(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model3(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model4(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model5(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model6(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model7(x), dim=1), dim=1))
        preds.append(torch.tensor(np.argmax(self.model8.Predict(self.device, self.model3, x), axis=1), device=self.device))
        preds.append(torch.tensor(np.argmax(self.model9.Predict(self.device, self.model2, x), axis=1), device=self.device))
        preds.append(torch.tensor(np.argmax(self.model10.Predict(self.device, self.model5, x), axis=1), device=self.device))
        
        votes = []
        for i in range(batchSize):
            classVotes = {}
            for c in preds:
                key = int(c[i].item())
                if key in classVotes:
                    classVotes[key] += 1
                else:
                    classVotes[key] = 1
            vote = max(classVotes, key=classVotes.get)
            votes.append(vote)
        
        return votes

def testing(device, modelCollection, dataLoaders):
    outputTotal = []
    targetTotal = []
    classCorrect = defaultdict(int)
    classTotal = defaultdict(int)

    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = modelCollection.Vote(data, len(data))
            outputTotal.extend(output)
            targetTotal.extend(target.cpu().numpy())

            for pred, true in zip(output, target.cpu().numpy()):
                classTotal[true] += 1
                if pred == true:
                    classCorrect[true] += 1

    overallAccuracy = accuracy_score(targetTotal, outputTotal)
    print(f"Overall Accuracy: {overallAccuracy:.2f}")

    print("Per-Class Accuracy:")
    for classID in sorted(classTotal.keys()):
        accuracy = 100.0 * classCorrect[classID] / classTotal[classID]
        print(f"Class {classID}: {accuracy:.2f}%")

    print("\nClassification Report:")
    print(classification_report(targetTotal, outputTotal))

    return 

def main():
    
    datasetName = input("Choose dataset[mnist/cifar10]: ")
    while not(datasetName == "mnist" or datasetName == "cifar10" or datasetName == "exit"):
        datasetName = input("Choose dataset[mnist/cifar10]: ")
    
    dataLoaders = pickle.load(open(f'../data/dataLoaders{datasetName}.pkl', 'rb'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model1Linear1 = torch.jit.load(f"../model1_Linear1/model1_Linear1_Scripted_{datasetName}.pt").to(device)
    model1Linear1.eval()
    
    model2CNN1 = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_{datasetName}.pt").to(device)
    model2CNN1.eval()
    
    model3CNN2 = torch.jit.load(f"../model3_CNN2/model3_CNN2_Scripted_{datasetName}.pt").to(device)
    model3CNN2.eval()
    
    model4CNN3 = torch.jit.load(f"../model4_CNN3/model4_CNN3_Scripted_{datasetName}.pt").to(device)
    model4CNN3.eval()
    
    model5CNN4 = torch.jit.load(f"../model5_CNN4/model5_CNN4_Scripted_{datasetName}.pt").to(device)
    model5CNN4.eval()
    
    model6CNN5 = torch.jit.load(f"../model6_CNN5/model6_CNN5_Scripted_{datasetName}.pt").to(device)
    model6CNN5.eval()
    
    model7CNN6 = torch.jit.load(f"../model7_CNN6/model7_CNN6_Scripted_{datasetName}.pt").to(device)
    model7CNN6.eval()
    
    model8SVM1 = pickle.load(open(f'../model8_SVM1_CNN2/model8_SVM1_{datasetName}.pkl', 'rb'))
    
    model9SVM2 = pickle.load(open(f'../model9_SVM2_CNN1/model9_SVM2_{datasetName}.pkl', 'rb'))
    
    model10SVM3 = pickle.load(open(f'../model10_SVM3_CNN4/model10_SVM3_{datasetName}.pkl', 'rb'))
    
    modelCollection = ModelCollection(
        device,
        model1Linear1,
        model2CNN1,
        model3CNN2,
        model4CNN3,
        model5CNN4,
        model6CNN5,
        model7CNN6,
        model8SVM1,
        model9SVM2,
        model10SVM3
        )
    
    testing(device, modelCollection, dataLoaders)

if __name__ == '__main__':
    main()