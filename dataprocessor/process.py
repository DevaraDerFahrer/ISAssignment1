import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import random_split, DataLoader
import pickle
    
inputSize = 32

#------------------MNIST------------------#

trainingTFMNIST = torchvision.transforms.Compose([
    transforms.Resize(inputSize),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(inputSize, padding=4),
    transforms.RandomRotation(10),
    ToTensor(),
    Normalize((0.5), (0.5))
    ])

testTFMNIST = torchvision.transforms.Compose([
    transforms.Resize(inputSize),
    ToTensor(),
    Normalize((0.5), (0.5))
    ])

rawDataMNIST = datasets.MNIST(
    root="../data",
    train= True,
    transform=None,
    download=True
)

testDataMNIST = datasets.MNIST(
    root="../data",
    train= False,
    transform=testTFMNIST,
    download=True
)

trainSizeMNIST = int(0.9 * len(rawDataMNIST))
validationSizeMNIST = len(rawDataMNIST) - trainSizeMNIST
trainSubsetMNIST, validationSubsetMNIST = random_split(rawDataMNIST, [trainSizeMNIST, validationSizeMNIST])

trainSubsetMNIST.dataset = datasets.MNIST(
    root="../data", 
    train=True, 
    transform=trainingTFMNIST
    )

validationSubsetMNIST.dataset = datasets.MNIST(
    root="../data", 
    train=True, 
    transform=testTFMNIST
    )

dataLoadersMNIST = {
    'train': DataLoader(
        trainSubsetMNIST,
        batch_size=100,
        shuffle=True,
        num_workers=1
    ),
    'validation': DataLoader(
        validationSubsetMNIST,
        batch_size=100,
        shuffle=True,
        num_workers=1
    ),
    'test': DataLoader(
        testDataMNIST,
        batch_size=100,
        shuffle=True,
        num_workers=1
    )
}

pickle.dump(dataLoadersMNIST, open(f'../data/dataLoadersMNIST.pkl', 'wb'))

#------------------CIFAR10------------------#

trainingTFCIFAR10 = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(inputSize, padding=4),
    transforms.RandomRotation(10),
    ToTensor(),
    Normalize((0.5), (0.5))
    ])

testTFCIFAR10 = torchvision.transforms.Compose([
    ToTensor(),
    Normalize((0.5), (0.5))
    ])

rawDataCIFAR10 = datasets.CIFAR10(
    root="../data",
    train= True,
    transform=trainingTFCIFAR10,
    download=True
)

testDataCIFAR10 = datasets.CIFAR10(
    root="../data",
    train= False,
    transform=testTFCIFAR10,
    download=True
)

trainSizeCIFAR10 = int(0.9 * len(rawDataCIFAR10))
validationSizeCIFAR10 = len(rawDataCIFAR10) - trainSizeCIFAR10
trainSubsetCIFAR10, validationSubsetCIFAR10 = random_split(rawDataCIFAR10, [trainSizeCIFAR10, validationSizeCIFAR10])

trainSubsetCIFAR10.dataset = datasets.CIFAR10(
    root="../data", 
    train=True, 
    transform=trainingTFCIFAR10
    )

validationSubsetCIFAR10.dataset = datasets.CIFAR10(
    root="../data", 
    train=True, 
    transform=testTFCIFAR10
    )

dataLoadersCIFAR10 = {
    'train': DataLoader(
        trainSubsetCIFAR10,
        batch_size=100,
        shuffle=True,
        num_workers=1
    ),
    'validation': DataLoader(
        validationSubsetCIFAR10,
        batch_size=100,
        shuffle=True,
        num_workers=1
    ),
    'test': DataLoader(
        testDataCIFAR10,
        batch_size=100,
        shuffle=True,
        num_workers=1
    )
}

pickle.dump(dataLoadersCIFAR10, open(f'../data/dataLoadersCIFAR10.pkl', 'wb'))
    
    