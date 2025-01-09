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

class cnnModel1(tNN.Module):
    def __init__(self, inputChannel, inputSize, numClassess):
        super(cnnModel1, self).__init__()
        self.outputChannel = 128
        self.conv1 = tNN.Conv2d(inputChannel, 128, kernel_size=5, stride=2, padding=2)
        self.conv2 = tNN.Conv2d(128, 128, kernel_size=3, padding="same")
        self.conv3 = tNN.Conv2d(128, 128, kernel_size=3, padding="same")
        self.conv4 = tNN.Conv2d(128, self.outputChannel, kernel_size=3, padding="same")
        self.convDrop = tNN.Dropout2d()
        self.fcDrop = tNN.Dropout(0.5)
        self.maxPool = tNN.MaxPool2d(2, 2)
        self.inputSize = ((inputSize - 5 + 2*2)//2) + 1 # 5x5 conv stride 2
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = int(self.inputSize * self.inputSize * self.outputChannel)
        self.fc1 = tNN.Linear(self.inputSize, 384)
        self.fc2 = tNN.Linear(384, numClassess)
   
    def forward(self, x):
        x = tNN.functional.relu(self.conv1(x))
        x = self.maxPool(x)
        x = tNN.functional.relu(self.conv2(x))
        x = self.maxPool(x)
        x = tNN.functional.relu(self.conv3(x))
        x = self.maxPool(x)
        x = tNN.functional.relu(self.conv4(x))
        x = self.maxPool(x)
        x = x.view(-1, self.inputSize)
        x = self.fc1(x)
        x = self.fcDrop(x)
        x = tNN.functional.relu(x)
        x = self.fc2(x)
        return x

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
    
    model2CNN = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_{datasetName}.pt").to(device)
    model2CNN.eval()
        
    correct = 0
    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = model2CNN(data)
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    testAccuracy = correct/len(dataLoaders['test'].dataset)
    print(f"Accuracy {testAccuracy} ({100. * correct / len(dataLoaders['test'].dataset):.0f}%)")
    return

if __name__ == '__main__':
    main()