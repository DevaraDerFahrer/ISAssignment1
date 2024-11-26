import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Grayscale
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as tNN
import torch.optim as tOptim
import time
import os
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def ExtractFeatures(device, model, dataLoader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataLoader:
            inputs = inputs.to(device)
            
            outputs = tNN.functional.relu(model.conv1(outputs))
            outputs = model.maxPool(outputs)
            outputs = tNN.functional.relu(model.conv2(outputs))
            outputs = model.maxPool(outputs)
            outputs = tNN.functional.relu(model.conv3(outputs))
            outputs = model.maxPool(outputs)
            outputs = tNN.functional.relu(model.conv4(outputs))
            outputs = model.maxPool(outputs)
            outputs = outputs.view(outputs.size(0), -1)

            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

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
        inputChannel = 1
        inputSize = 32
        numClassess = 10
        
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
        
    elif datasetName == "cifar10":
        inputChannel = 3
        inputSize = 32
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnnModel = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_{datasetName}.pt").to(device)
    cnnModel.eval()

    extractedFeatures = ExtractFeatures(cnnModel)

    trainFeatures, trainLabels = ExtractFeatures(device, cnnModel, trainData)
    testFeatures, testLabels = ExtractFeatures(device, cnnModel, testData)

    # Train SVM classifier
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(trainFeatures, trainLabels)

    # Validate SVM classifier
    predictions = svm.predict(testFeatures)
    accuracy = accuracy_score(testLabels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    #print(classification_report(val_labels, val_predictions, target_names=train_dataset.classes))


if __name__ == '__main__':
    main()