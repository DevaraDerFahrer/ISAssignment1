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

def main():
    
    programStartTime = time.time()

    datasetName = input("Choose dataset[mnist/cifar10]: ")
    while not(datasetName == "mnist" or datasetName == "cifar10" or datasetName == "exit"):
        datasetName = input("Choose dataset[mnist/cifar10]: ")

    testData = any
    
    if datasetName == "exit":
        return
    elif datasetName == "mnist":
        inputChannel = 1
        inputSize = 32
        numClassess = 10
        
        testTF = torchvision.transforms.Compose([
            transforms.Resize(inputSize),
            ToTensor(),
            Normalize((0.5), (0.5))
            ])

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
        
        testTF = torchvision.transforms.Compose([
            ToTensor(),
            Normalize((0.5), (0.5))
            ])
        
        testData = datasets.CIFAR10(
            root="../data",
            train= False,
            transform=testTF,
            download=True
        )

    else:
        return
    
    
    dataLoaders = {
        'test': DataLoader(
            testData,
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnnModel = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_{datasetName}.pt").to(device)
    cnnModel.eval()
    
    model3SVM = pickle.load(open(f'../model3_SVM1/model3_SVM1_{datasetName}.pkl', 'rb'))

    predictions = []
    targets = []
    for data, target in dataLoaders['test']:
        data, target = data.to(device), target.to(torch.device('cpu'))
        output = model3SVM.Predict(device, cnnModel, data)
        predicted_classes = np.argmax(output, axis=1)
        predictions.extend(predicted_classes)
        targets.extend(target.numpy())

    accuracy = accuracy_score(predictions, targets)
    print(accuracy)
    
    return 
    
if __name__ == '__main__':
    main()