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
    def __init__(self, device, model1, model2, model3):
        self.device = device
        self.numOfModels = 2
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
    def GetOutputFromModels(self, x):
        outputs = []
        #outputs.append(torch.softmax(self.model1(x), dim=1))
        outputs.append(torch.softmax(self.model2(x), dim=1))
        outputs.append(torch.tensor(self.model3.Predict(self.device, self.model2, x), device=self.device).float())
        return torch.cat((outputs), dim=1)

def training(device, model, modelCollection, dataLoaders, criterion, optimizer, epoch):
    model.train()
    for batchIDx, (data,  target) in enumerate(dataLoaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(modelCollection.GetOutputFromModels(data))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batchIDx % 20 == 0:
            print(f"Train epoch: {epoch} [{batchIDx * len(data)}/{len(dataLoaders['train'].dataset)} ({100. * batchIDx / len(dataLoaders['train']):.0f}%)]\t{loss.item():.6f}")

def testing(device, model, modelCollection, dataLoaders, criterion):
    model.eval()
    testLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(modelCollection.GetOutputFromModels(data))
            testLoss += criterion(output, target).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(dataLoaders['test'].dataset)
    testAccuracy = correct/len(dataLoaders['test'].dataset)
    print(f"\nTest set: Average loss: {testLoss:.4f}, Accuracy {correct}/{len(dataLoaders['test'].dataset)} ({100. * correct / len(dataLoaders['test'].dataset):.0f}%)\n")
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
    
    model1Linear = torch.jit.load(f"../model1_Linear1/model1_Linear1_Scripted_{datasetName}.pt").to(device)
    model1Linear.eval()
    
    model2CNN = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_{datasetName}.pt").to(device)
    model2CNN.eval()
    
    model3SVM = pickle.load(open(f'../model3_SVM1/model3_SVM1_{datasetName}.pkl', 'rb'))
    
    modelCollection = ModelCollection(device, model1Linear, model2CNN, model3SVM)
    
    model = ensembleModel(device, modelCollection.numOfModels, numClassess).to(device)
    criterion = tNN.CrossEntropyLoss().to(device)
    optimizer = tOptim.Adam(model.parameters(),lr=0.001)
    
    numOfEpoch = 20
    
    losses = []
    accuracies = []
    
    for epoch in range(1, numOfEpoch+1):
        training(device, model, modelCollection, dataLoaders, criterion, optimizer, epoch)
        testLoss, testAccuracy = testing(device, model, modelCollection, dataLoaders, criterion)
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
    
    torch.save(model, f'model1_Linear1_{datasetName}.pt')
    modelScripted = torch.jit.script(model)
    modelScripted.save(f'model1_Linear1_Scripted_{datasetName}.pt')
    print(f"model saved, elapsed time: {time.time() - programStartTime}")

if __name__ == '__main__':
    main()