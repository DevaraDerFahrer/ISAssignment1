import torch
import time
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Grayscale
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as tNN
import torch.optim as tOptim
import time

class cnnModel2RGB(tNN.Module):
    def __init__(self, inputChannel, inputSize, numClassess):
        super(cnnModel2RGB, self).__init__()
        self.outputChannel = 256
        self.conv1 = tNN.Conv2d(inputChannel, 64, kernel_size=3)
        self.conv2 = tNN.Conv2d(64, 128, kernel_size=3)
        self.conv3 = tNN.Conv2d(128, self.outputChannel, kernel_size=3)
        self.convDrop = tNN.Dropout2d()
        self.maxPool = tNN.MaxPool2d(2, 2)
        self.inputSize = (inputSize/2/2/2) * self.outputChannel
        self.fc1 = tNN.Linear(self.inputSize,500)
        self.fc2 = tNN.Linear(500,numClassess)
   
    def forward(self, x):
        x = self.conv1(x)
        x = tNN.functional.relu(self.maxPool(x))
        x = self.conv2(x)
        x = tNN.functional.relu(self.maxPool(x))
        x = self.conv3(x)
        x = tNN.functional.relu(self.maxPool(x))
        x = self.convDrop(x)
        x = x.view(-1, self.inputSize)
        x = tNN.functional.relu(self.fc1(x))
        x = tNN.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return tNN.functional.softmax(x)

def training(device, model, dataLoaders, criterion, optimizer, epoch):
    model.train()
    for batchIDx, (data,  target) in enumerate(dataLoaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batchIDx % 20 == 0:
            print(f"Train epoch: {epoch} [{batchIDx * len(data)}/{len(dataLoaders['train'].dataset)} ({100. * batchIDx / len(dataLoaders['train']):.0f}%)]\t{loss.item():.6f}")

def testing(device, model, dataLoaders, criterion):
    model.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += criterion(output, target).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(dataLoaders['test'].dataset)
    print(f"\nTest set: Average loss: {testLoss:.4f}, Accuracy {correct}/{len(dataLoaders['test'].dataset)} ({100. * correct / len(dataLoaders['test'].dataset):.0f}%)\n")

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
        inputSize = 28
        numClassess = 10
        
        trainData = datasets.MNIST(
            root="../data",
            train= True,
            transform=ToTensor(),
            download=True
        )

        testData = datasets.MNIST(
            root="../data",
            train= False,
            transform=ToTensor(),
            download=True
        )
        
    elif datasetName == "cifar10":
        inputSize = 32
        numClassess = 10
        
        trainData = datasets.CIFAR10(
            root="../data",
            train= True,
            transform=ToTensor(),
            download=True
        )
        
        testData = datasets.CIFAR10(
            root="../data",
            train= False,
            transform=ToTensor(),
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
    model = cnnModel2RGB(3, inputSize, numClassess).to(device)
    criterion = tNN.CrossEntropyLoss().to(device)
    optimizer = tOptim.Adam(model.parameters(),lr=0.001)
    
    for epoch in range(20):
        training(device, model, dataLoaders, criterion, optimizer, epoch + 1)
        testing(device, model, dataLoaders, criterion)
        
    modelScripted = torch.jit.script(model)
    modelScripted.save(f'model3_CNN2RGB_Scripted_{datasetName}.pt')
    print("model saved")

if __name__ == '__main__':
    main()