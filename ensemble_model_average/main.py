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
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import mylibrary
    
class ModelCollection:
    def __init__(self, device, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10):
        self.device = device
        self.numOfModels = 10
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
    def Average(self, x, batchSize, weights):
        probs = [
            torch.softmax(self.model1(x), dim=1).cpu().numpy(),
            torch.softmax(self.model2(x), dim=1).cpu().numpy(),
            torch.softmax(self.model3(x), dim=1).cpu().numpy(),
            torch.softmax(self.model4(x), dim=1).cpu().numpy(),
            torch.softmax(self.model5(x), dim=1).cpu().numpy(),
            torch.softmax(self.model6(x), dim=1).cpu().numpy(),
            torch.softmax(self.model7(x), dim=1).cpu().numpy(),
            self.model8.Predict(self.device, self.model3, x),
            self.model9.Predict(self.device, self.model2, x),
            self.model10.Predict(self.device, self.model5, x)
        ]
        
        finalPreds = []
        
        for i in range(batchSize):
            classProbs = [0,0,0,0,0,0,0,0,0,0]
            for c in range(10):
                classProbsTotal = 0
                for m in range(10):
                    classProbsTotal += probs[m][i][c] * weights[m][c]
                classProbs[c] = classProbsTotal
            maxV = max(classProbs)
            maxC = classProbs.index(maxV)
            finalPreds.append(maxC)
            
        return finalPreds
    
def testing(device, modelCollection, dataLoaders, weights):
    preds = []
    targets = []
    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            preds.extend(modelCollection.Average(data, len(data), weights))
            targets.extend(target.cpu().numpy())

    return preds, targets

def testingAll(device, modelCollection, dataLoaders, weights):
    preds = []
    targets = []
    classCorrect = defaultdict(int)
    classTotal = defaultdict(int)

    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            batchPreds = modelCollection.Average(data, len(data), weights)
            preds.extend(batchPreds)
            targets.extend(target.cpu().numpy())

            for pred, true in zip(batchPreds, target.cpu().numpy()):
                classTotal[true] += 1
                if pred == true:
                    classCorrect[true] += 1

    overallAccuracy = accuracy_score(targets, preds)

    perClassAccuracy = {
        classID: 100.0 * classCorrect[classID] / classTotal[classID]
        for classID in sorted(classTotal.keys())
    }

    classReport = classification_report(targets, preds)

    return overallAccuracy, perClassAccuracy, classReport

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
    
    weights = pickle.load(open(f'weights_{datasetName}.pkl', 'rb'))
    
    overallAccuracy, perClassAccuracy, classReport = testingAll(device, modelCollection, dataLoaders, weights)
    
    print(f"Overall Accuracy: {overallAccuracy:.2f}")
    print("\nPer-Class Accuracy:")
    for class_id, accuracy in perClassAccuracy.items():
        print(f"Class {class_id}: {accuracy:.2f}%")
    
    print("\nClassification Report:")
    print(classReport)
    
if __name__ == '__main__':
    main()