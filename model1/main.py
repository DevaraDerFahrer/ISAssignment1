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

dataLoaders = {
    'train': DataLoader(
        mnistTrainData,
        batch_size=100,
        shuffle=False,
        num_workers=1
    )
}

class model1Linear(tNN.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = tNN.Linear(784,36)
        self.fc2 = tNN.Linear(36,36)
        self.out = tNN.Linear(36,10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = tNN.functional.relu(self.fc1(x))
        x = tNN.functional.relu(self.fc2(x))
        x = self.out(x)
        return tNN.functional.softmax(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model1Linear().to(device)
criterion = tNN.CrossEntropyLoss()
optimizer = tOptim.Adam(model.parameters(),lr=0.001)

def training(epoch):
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

if __name__ == '__main__':
    for epoch in range(1,11):
        training(epoch)
    torch.save(model, "trainedModel.pt")
    print("model saved")