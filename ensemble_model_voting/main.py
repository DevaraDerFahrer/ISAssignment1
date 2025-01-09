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
import time
import os

def SavePlotAsVectors(x, y, title, xlabel, ylabel, filename, output_dir="vector_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{filename}.svg"), format="svg", dpi=300)
    plt.close()
    print(f"Saved vector image: {os.path.join(output_dir, f'{filename}.svg')}")

class modelCNNSVM:
    def __init__(self, svmModel):
        self.SVMModel = svmModel
        
    def ExtractFeatures(self, device, cnnModel, dataLoader):
        features = []
        with torch.no_grad():
            for inputs in dataLoader:
                inputs = inputs.to(device)

                outputs = tNN.functional.relu(cnnModel.conv1(inputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = tNN.functional.relu(cnnModel.conv2(outputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = tNN.functional.relu(cnnModel.conv3(outputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = tNN.functional.relu(cnnModel.conv4(outputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = outputs.view(-1, cnnModel.inputSize)
                
                batch_features = outputs.view(outputs.size(0), -1).cpu().numpy()
                features.extend(batch_features)

        return features
    
    def Predict(self, device, cnnModel, dataLoader):
        features = self.ExtractFeatures(device, cnnModel, dataLoader)
        return self.SVMModel.predict_proba(features)

class ensembleModel(tNN.Module):
    def __init__(self, device, numOfModels, numClasses):
        super(ensembleModel, self).__init__()
        self.inputSize = numOfModels * numClasses
        self.out = tNN.Linear(self.inputSize,numClasses)
        self.device = device
        self.layerNorm = tNN.LayerNorm(self.inputSize)
    def forward(self, x):
        x = x.view(-1, self.inputSize)
        x = self.layerNorm(x)
        x = self.out(x)
        return tNN.functional.softmax(x, dim=1)
    
class ModelCollection:
    def __init__(self, device, model1, model2, model3, model4, model5, model6, model7):
        self.device = device
        self.numOfModels = 2
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
    def GetOutputFromModels(self, x):
        outputs = []
        #outputs.append(torch.softmax(self.model1(x), dim=1))
        outputs.append(torch.softmax(self.model2(x), dim=1))
        outputs.append(torch.tensor(self.model3.Predict(self.device, self.model2, x), device=self.device).float())
        return torch.cat((outputs), dim=1)
    def Vote(self, x, batchSize):
        preds = []
        preds.append(torch.argmax(torch.softmax(self.model1(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model2(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model3(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model4(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model5(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model6(x), dim=1), dim=1))
        preds.append(torch.argmax(torch.softmax(self.model7(x), dim=1), dim=1))
        
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
    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = modelCollection.Vote(data, len(data))
            outputTotal.extend(output)
            targetTotal.extend(target.cpu().numpy())
    
    print(accuracy_score(outputTotal, targetTotal))

    return 

def main():
    programStartTime = time.time()

    datasetName = input("Choose dataset[mnist/cifar10]: ")
    while not(datasetName == "mnist" or datasetName == "cifar10" or datasetName == "exit"):
        datasetName = input("Choose dataset[mnist/cifar10]: ")
    
    trainData = any
    testData = any    
    
    if datasetName == "exit":
        return
    elif datasetName == "mnist":
        inputSize = 32
        inputChannel = 1
        numClassess = 10
        
        trainingTF = torchvision.transforms.Compose([
            transforms.Resize(inputSize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(inputSize, padding=4),
            transforms.RandomRotation(10),
            ToTensor(),
            Normalize((0.5), (0.5))
            ])
        
        testTF = torchvision.transforms.Compose([
            transforms.Resize(inputSize),
            ToTensor(),
            Normalize((0.5), (0.5))
            ])
        
        trainData = datasets.MNIST(
            root="../data",
            train= True,
            transform=trainingTF,
            download=True
        )

        testData = datasets.MNIST(
            root="../data",
            train= False,
            transform=testTF,
            download=True
        )
        
    elif datasetName == "cifar10":
        inputSize = 32
        inputChannel = 3
        numClassess = 10
        
        trainingTF = torchvision.transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(inputSize, padding=4),
            transforms.RandomRotation(10),
            ToTensor(),
            Normalize((0.5), (0.5))
            ])
        
        testTF = torchvision.transforms.Compose([
            ToTensor(),
            Normalize((0.5), (0.5))
            ])
        
        trainData = datasets.CIFAR10(
            root="../data",
            train= True,
            transform=trainingTF,
            download=True
        )
        
        testData = datasets.CIFAR10(
            root="../data",
            train= False,
            transform=testTF,
            download=True
        )

    else:
        return
    
    dataLoaders = {
        'train': DataLoader(
            trainData,
            batch_size=100,
            shuffle=True,
            num_workers=1
        ),
        'test': DataLoader(
            testData,
            batch_size=100,
            shuffle=True,
            num_workers=1
        )
    }
    
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
    
    modelCollection = ModelCollection(
        device, 
        model1Linear1, 
        model2CNN1, 
        model3CNN2,
        model4CNN3,
        model5CNN4,
        model6CNN5,
        model7CNN6
        )
    
    testing(device, modelCollection, dataLoaders)

if __name__ == '__main__':
    main()