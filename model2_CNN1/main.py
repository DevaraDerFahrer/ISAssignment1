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

class cnnModel1(tNN.Module):
    def __init__(self, inputChannel, inputSize, numClassess):
        super(cnnModel1, self).__init__()
        self.conv1 = tNN.Conv2d(inputChannel, 10, kernel_size=5)
        self.outputChannel = 20
        self.conv2 = tNN.Conv2d(10, self.outputChannel, kernel_size=5)
        self.conv2Drop = tNN.Dropout2d()
        self.inputSize = ((inputSize - 5 + 2*0)/1) + 1 # 1st convolution
        self.inputSize = ((self.inputSize - 2 + 2*0)/2) + 1 # maxpool 2x2 stride 2
        self.inputSize = ((self.inputSize - 5 + 2*0)/1) + 1 # 2nd convolution
        self.inputSize = ((self.inputSize - 2 + 2*0)/2) + 1 # maxpool 2x2 stride 2
        self.inputSize = int(self.inputSize * self.inputSize * self.outputChannel)
        self.fc1 = tNN.Linear(self.inputSize,50)
        self.fc2 = tNN.Linear(50,numClassess)
   
    def forward(self, x):
        x = self.conv1(x)
        x = tNN.functional.relu(tNN.functional.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.conv2Drop(x)
        x = tNN.functional.relu(tNN.functional.max_pool2d(x, 2))
        x = x.view(-1, self.inputSize)
        x = tNN.functional.relu(self.fc1(x))
        x = tNN.functional.dropout(x, training=self.training)
        x = self.fc2(x)
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
    
    inputSize = 28
    inputChannel = 1
    numClassess = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnnModel1(inputChannel, inputSize, numClassess).to(device)
    criterion = tNN.CrossEntropyLoss().to(device)
    optimizer = tOptim.Adam(model.parameters(),lr=0.001)
    
    for epoch in range(50):
        training(device, model, criterion, optimizer, epoch + 1)
        testing(device, model, criterion)
        
    modelScripted = torch.jit.script(model)
    modelScripted.save('model2_CNN1_Scripted.pt')
    print("model saved")

if __name__ == '__main__':
    main()