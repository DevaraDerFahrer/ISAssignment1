import torch
import time
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Grayscale
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as tNN
import torch.optim as tOptim
import time
import os

class linearModel1(tNN.Module):
    def __init__(self, inputSize, inputChannel, numClasses):
        super(linearModel1, self).__init__()
        self.inputSize = inputSize * inputSize * inputChannel
        self.fc1 = tNN.Linear(self.inputSize,300)
        self.fc2 = tNN.Linear(300,300)
        self.out = tNN.Linear(300,numClasses)
    def forward(self, x):
        x = x.view(-1, self.inputSize)
        x = tNN.functional.relu(self.fc1(x))
        x = tNN.functional.relu(self.fc2(x))
        x = self.out(x)
        return tNN.functional.softmax(x, dim=1)

def main():
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataLoaders = {
        'test': DataLoader(
            testData,
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
    }
    
    model1Linear = torch.jit.load(f"../model1_Linear1/model1_Linear1_Scripted_{datasetName}.pt").to(device)
    model1Linear.eval()
        
    correct = 0
    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = model1Linear(data)
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    testAccuracy = correct/len(dataLoaders['test'].dataset)
    print(f"Accuracy {testAccuracy} ({100. * correct / len(dataLoaders['test'].dataset):.0f}%)")
    return

if __name__ == '__main__':
    main()