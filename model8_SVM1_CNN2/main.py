import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Grayscale
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader
import torch.nn as tNN
import torch.optim as tOptim
import time
import os
import sys
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np

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
        labels = []
        with torch.no_grad():
            for inputs, target in dataLoader:
                inputs = inputs.to(device)

                outputs = tNN.functional.relu(cnnModel.conv1(inputs))
                outputs = tNN.functional.relu(cnnModel.conv2(outputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = tNN.functional.relu(cnnModel.conv3(outputs))
                outputs = tNN.functional.relu(cnnModel.conv4(outputs))
                outputs = tNN.functional.relu(cnnModel.conv4(outputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = outputs.view(-1, cnnModel.inputSize)
                
                batch_features = outputs.view(outputs.size(0), -1).cpu().numpy()
                features.extend(batch_features)
                labels.extend(target.cpu().numpy())

        return features, labels
    
    def Predict(self, device, cnnModel, dataLoader):
        features, _ = self.ExtractFeatures(device, cnnModel, dataLoader)
        return self.SVMModel.predict_proba(features)
    
def ExtractFeatures(device, model, dataLoader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, target in dataLoader:
            inputs = inputs.to(device)
            
            outputs = tNN.functional.relu(model.conv1(inputs))
            outputs = tNN.functional.relu(model.conv2(outputs))
            outputs = model.maxPool(outputs)
            outputs = tNN.functional.relu(model.conv4(outputs))
            outputs = tNN.functional.relu(model.conv3(outputs))
            outputs = tNN.functional.relu(model.conv5(outputs))
            outputs = model.maxPool(outputs)
            outputs = outputs.view(-1, model.inputSize)

            batch_features = outputs.view(outputs.size(0), -1).cpu().numpy()
            features.extend(batch_features)
            labels.extend(target.cpu().numpy())
            
    return features, labels

def preprocessMNIST():
    inputSize = 32
    
    trainingTF = torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(inputSize),
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
    
    dataLoaders = {
        'train': DataLoader(
            trainData,
            shuffle=True,
            num_workers=4
        ),
        'test': DataLoader(
            testData,
            shuffle=True,
            num_workers=4
        )
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnnModel = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_mnist.pt").to(device)
    cnnModel.eval()

    trainFeatures, trainLabels = ExtractFeatures(device, cnnModel, dataLoaders['train'])
    testFeatures, testLabels = ExtractFeatures(device, cnnModel, dataLoaders['test'])
    
    pickle.dump(trainFeatures, open('trainFeatures_mnist.pkl', 'wb'))
    pickle.dump(trainLabels, open('trainLabels_mnist.pkl', 'wb'))
    pickle.dump(testFeatures, open('testFeatures_mnist.pkl', 'wb'))
    pickle.dump(testLabels, open('testLabels_mnist.pkl', 'wb'))
    
    return trainFeatures, trainLabels, testFeatures, testLabels

def preprocessCIFAR10():
    inputSize = 32
    
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
    
    dataLoaders = {
        'train': DataLoader(
            trainData,
            shuffle=True,
            num_workers=4
        ),
        'test': DataLoader(
            testData,
            shuffle=True,
            num_workers=4
        )
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnnModel = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_cifar10.pt").to(device)
    cnnModel.eval()

    trainFeatures, trainLabels = ExtractFeatures(device, cnnModel, dataLoaders['train'])
    testFeatures, testLabels = ExtractFeatures(device, cnnModel, dataLoaders['test'])
    
    pickle.dump(trainFeatures, open('trainFeatures_cifar10.pkl', 'wb'))
    pickle.dump(trainLabels, open('trainLabels_cifar10.pkl', 'wb'))
    pickle.dump(testFeatures, open('testFeatures_cifar10.pkl', 'wb'))
    pickle.dump(testLabels, open('testLabels_cifar10.pkl', 'wb'))
    
    return trainFeatures, trainLabels, testFeatures, testLabels

def main():
    
    programStartTime = time.time()

    datasetName = input("Choose dataset[mnist/cifar10]: ")
    while not(datasetName == "mnist" or datasetName == "cifar10" or datasetName == "exit"):
        datasetName = input("Choose dataset[mnist/cifar10]: ")
    
    trainFeatures = any
    trainLabels = any
    testFeatures = any
    testLabels = any
   
    if datasetName == "exit":
        return
    elif datasetName == "mnist":
        try:
            trainFeatures = pickle.load(open('trainFeatures_mnist.pkl', 'rb'))
            trainLabels = pickle.load(open('trainLabels_mnist.pkl', 'rb'))
            testFeatures = pickle.load(open('testFeatures_mnist.pkl', 'rb'))
            testLabels = pickle.load(open('testLabels_mnist.pkl', 'rb'))
        except:
           trainFeatures, trainLabels, testFeatures, testLabels = preprocessMNIST()
        
    elif datasetName == "cifar10":
        try:
            trainFeatures = pickle.load(open('trainFeatures_cifar10.pkl', 'rb'))
            trainLabels = pickle.load(open('trainLabels_cifar10.pkl', 'rb'))
            testFeatures = pickle.load(open('testFeatures_cifar10.pkl', 'rb'))
            testLabels = pickle.load(open('testLabels_cifar10.pkl', 'rb'))
        except:
           trainFeatures, trainLabels, testFeatures, testLabels = preprocessCIFAR10()
           
    else:
        return
    
    param_grid = {
        'C': [0.1],#[0.1, 1, 10],
        'kernel':  ['linear'],#['linear', 'rbf', 'poly'],
        'gamma': [0.001]#[0.001, 0.01, 0.1]
    }
    classifier = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    classifier.fit(trainFeatures, trainLabels)
    bestModel = classifier.best_estimator_
    
    predLabels = bestModel.predict(testFeatures)
    accuracy = accuracy_score(testLabels, predLabels)
    print(accuracy)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnnModel = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_{datasetName}.pt").to(device)
    cnnModel.eval()
    packagedModel = modelCNNSVM(bestModel)
    pickle.dump(packagedModel, open(f'model3_SVM1_{datasetName}.pkl', 'wb'))
    pickle.dump(bestModel, open(f'model3_SVM1_unpackaged_{datasetName}.pkl', 'wb'))
    print(f"model saved, elapsed time: {time.time() - programStartTime}")

if __name__ == '__main__':
    main()