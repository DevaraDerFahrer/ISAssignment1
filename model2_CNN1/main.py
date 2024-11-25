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
    testAccuracy = correct/len(dataLoaders['test'].dataset)
    print(f"\nTest set: Average loss: {testLoss:.4f}, Accuracy {testAccuracy} ({100. * correct / len(dataLoaders['test'].dataset):.0f}%)\n")
    return testLoss, testAccuracy

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
        inputSize = 28
        numClassess = 10
        
        trainingTF = torchvision.transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        testTF = torchvision.transforms.Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        testTF = torchvision.transforms.Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    model = cnnModel1(inputChannel, inputSize, numClassess).to(device)
    criterion = tNN.CrossEntropyLoss().to(device)
    optimizer = tOptim.Adam(model.parameters(), lr=0.001)
    
    dataLoaders = {
        'train': DataLoader(
            trainData,
            batch_size=128,
            shuffle=True,
            num_workers=4
        ),
        'test': DataLoader(
            testData,
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
    }
    
    numOfEpoch = 100
    
    losses = []
    accuracies = []
    
    for epoch in range(1, numOfEpoch+1):
        training(device, model, dataLoaders, criterion, optimizer, epoch)
        testLoss, testAccuracy =  testing(device, model, dataLoaders, criterion)
        losses.append(testLoss)
        accuracies.append(testAccuracy)
        
    SavePlotAsVectors(
        x=range(1, numOfEpoch+1), y=losses,
        title="Training Loss Over Epochs",
        xlabel="Epochs", ylabel="Loss",
        filename="training_loss",
        output_dir="results"
    )

    SavePlotAsVectors(
        x=range(1, numOfEpoch+1), y=accuracies,
        title="Test Accuracy Over Epochs",
        xlabel="Epochs", ylabel="Accuracy (%)",
        filename="test_accuracy",
        output_dir="results"
    )
        
    torch.save(model, f'model2_CNN1_{datasetName}.pt')
    modelScripted = torch.jit.script(model)
    modelScripted.save(f'model2_CNN1_Scripted_{datasetName}.pt')
    print(f"model saved, elapsed time: {time.time() - programStartTime}")

if __name__ == '__main__':
    main()