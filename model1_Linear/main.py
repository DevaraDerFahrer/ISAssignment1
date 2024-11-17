import torch
import time
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as tNN
import torch.optim as tOptim
import time

programStartTime = time.time()

mnistTrainData = datasets.MNIST(
    root="../data",
    train= True,
    transform=ToTensor(),
    download=True
)

mnistTestingData = datasets.MNIST(
    root="../data",
    train= False,
    transform=ToTensor(),
    download=True
)

dataLoaders = {
    'train': DataLoader(
        mnistTrainData,
        batch_size=100,
        shuffle=True,
        num_workers=1
    ),
    'test': DataLoader(
        mnistTestingData,
        batch_size=100,
        shuffle=True,
        num_workers=1
    )
}

class linearModel1(tNN.Module):
    def __init__(self, inputSize, numClasses):
        super(linearModel1, self).__init__()
        self.fc1 = tNN.Linear(inputSize,36)
        self.fc2 = tNN.Linear(36,36)
        self.out = tNN.Linear(36,numClasses)
    def forward(self, x):
        x = x.view(-1, 784)
        x = tNN.functional.relu(self.fc1(x))
        x = tNN.functional.relu(self.fc2(x))
        x = self.out(x)
        return tNN.functional.softmax(x)

def training(device, model, criterion, optimizer, epoch):
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

def testing(device, model, criterion):
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
    
    inputSize = 28*28
    numClassess = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = linearModel1(inputSize, numClassess).to(device)
    criterion = tNN.CrossEntropyLoss().to(device)
    optimizer = tOptim.Adam(model.parameters(),lr=0.001)
    
    for epoch in range(50):
        training(device, model, criterion, optimizer, epoch)
        testing(device, model, criterion)
    
    modelScripted = torch.jit.script(model)
    modelScripted.save('model1_Linear_Scripted.pt')
    print(f"model saved, elapsed time: {time.time() - programStartTime}")

if __name__ == '__main__':
    main()