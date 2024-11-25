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
    
    trainingTF = torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(10),
        ToTensor(),
        Normalize((0.5), (0.5))
        ])
        
    testTF = torchvision.transforms.Compose([
        ToTensor(),
        Normalize((0.5), (0.5))
        ])    
    
    if datasetName == "exit":
        return
    elif datasetName == "mnist":
        inputSize = 28
        inputChannel = 1
        numClassess = 10
        
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
    model = linearModel1(inputSize, inputChannel, numClassess).to(device)
    criterion = tNN.CrossEntropyLoss().to(device)
    optimizer = tOptim.Adam(model.parameters(),lr=0.001)
    
    numOfEpoch = 100
    
    losses = []
    accuracies = []
    
    for epoch in range(numOfEpoch):
        training(device, model, dataLoaders, criterion, optimizer, epoch)
        testing(device, model, dataLoaders, criterion)
    
    SavePlotAsVectors(
        x=range(1, numOfEpoch+1), y=losses,
        title="Training Loss Over Epochs",
        xlabel="Epochs", ylabel="Loss",
        filename=f"training_loss_{datasetName}",
        output_dir="results"
    )

    SavePlotAsVectors(
        x=range(1, numOfEpoch+1), y=accuracies,
        title="Test Accuracy Over Epochs",
        xlabel="Epochs", ylabel="Accuracy (%)",
        filename=f"test_accuracy_{datasetName}",
        output_dir="results"
    )    
    
    torch.save(model, f'model1_Linear1_{datasetName}.pt')
    modelScripted = torch.jit.script(model)
    modelScripted.save(f'model1_Linear1_Scripted_{datasetName}.pt')
    print(f"model saved, elapsed time: {time.time() - programStartTime}")

if __name__ == '__main__':
    main()