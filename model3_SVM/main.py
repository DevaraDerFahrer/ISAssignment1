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

def ExtractFeatures(device, model, dataLoader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, target in dataLoader:
            inputs = inputs.to(device)
            
            outputs = tNN.functional.relu(model.conv1(inputs))
            outputs = model.maxPool(outputs)
            outputs = tNN.functional.relu(model.conv2(outputs))
            outputs = model.maxPool(outputs)
            outputs = tNN.functional.relu(model.conv3(outputs))
            outputs = model.maxPool(outputs)
            outputs = tNN.functional.relu(model.conv4(outputs))
            outputs = model.maxPool(outputs)
            outputs = outputs.view(outputs.size(0), -1)

            features.append(outputs)
            labels.append(target)
            
    return torch.cat(features), torch.cat(labels)

class model3_SVM1(tNN.Module):
    def __init__(self, inputDimension, numClasses):
        super(model3_SVM1, self).__init__()
        self.weights = tNN.Parameter(torch.randn(inputDimension, numClasses))
        self.bias = tNN.Parameter(torch.zeros(numClasses))

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias

def HingeLossFunction(device, model, labels, inputs, numClasses=10, regularization=0.001):
    oneHotLabels = torch.nn.functional.one_hot(labels, num_classes=numClasses).float()
    oneHotLabels = oneHotLabels.to(device)
    inputs = inputs.to(device)
    margins = 1 - oneHotLabels * inputs
    print("oit")
    hinge = torch.clamp(margins, min=0).sum(dim=1).mean()
    regTerm = regularization * torch.norm(model.weights) ** 2
    return hinge + regTerm


def training(device, model, labels, features, criterion, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(device, model, labels, outputs)
    loss.backward()
    optimizer.step()
    print(f"Train epoch: {epoch} \t {loss.item():.6f}")

def testing(device, model, criterion, testLabels, testFeatures):
    model.eval()
    
    with torch.no_grad():
        outputs = model(testFeatures)
        predictions = torch.argmax(outputs, dim=1)
        testAccuracy = (predictions == testLabels).float().mean().item()
        testLoss = criterion(device, model, testLabels, outputs)
    
    print(f"\nTest set: Loss: {testLoss.item()*100}, Accuracy {testAccuracy*100}\n")
    return testLoss.item(), testAccuracy

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnnModel = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_{datasetName}.pt").to(device)
    cnnModel.eval()

    trainFeatures, trainLabels = ExtractFeatures(device, cnnModel, dataLoaders['train'])
    testFeatures, testLabels = ExtractFeatures(device, cnnModel, dataLoaders['test'])
    
    trainFeatures = trainFeatures.to(device)
    trainLabels = trainLabels.to(device)
    testFeatures = testFeatures.to(device)
    testLabels = testLabels.to(device)

    inputDimension = trainFeatures.shape[1]
    model = model3_SVM1(128, numClassess).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    numOfEpoch = 100
    
    losses = []
    accuracies = []

    for epoch in range(1, numOfEpoch+1):
        criterion = HingeLossFunction
        training(device, model, trainLabels, trainFeatures, criterion, optimizer, epoch)
        testLoss, testAccuracy = testing(device, model, criterion, testFeatures, testLabels)
        losses.append(testLoss)
        accuracies.append(testAccuracy)
        
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

    torch.save(model, f'model3_SVM1_{datasetName}.pt')
    modelScripted = torch.jit.script(model)
    modelScripted.save(f'model3_SVM1_Scripted_{datasetName}.pt')
    print(f"model saved, elapsed time: {time.time() - programStartTime}")

if __name__ == '__main__':
    main()