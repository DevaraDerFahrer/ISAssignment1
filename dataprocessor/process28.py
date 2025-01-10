import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import random_split, DataLoader
import pickle
    
inputSize = 28

#------------------MNIST------------------#

trainingTFMNIST = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(inputSize, padding=4),
    transforms.RandomRotation(10),
    ToTensor(),
    Normalize((0.5), (0.5))
    ])

testTFMNIST = torchvision.transforms.Compose([
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

pickle.dump(dataLoadersMNIST, open(f'../data/dataLoadersMNIST28.pkl', 'wb'))