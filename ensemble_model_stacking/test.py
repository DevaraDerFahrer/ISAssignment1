import torch
import time
import pickle
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
import torch.nn as tNN
import torch.optim as tOptim
import time
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import mylibrary

class ensembleModel(tNN.Module):
    def __init__(self, device, numOfModels, numClasses):
        super(ensembleModel, self).__init__()
        self.inputSize = numOfModels * numClasses
        self.out = tNN.Linear(self.inputSize,numClasses)
        self.fcDrop = tNN.Dropout(0.5)
        self.device = device
        self.layerNorm = tNN.LayerNorm(self.inputSize)
    def forward(self, x):
        x = x.view(-1, self.inputSize)
        x = self.layerNorm(x)
        x = tNN.functional.relu(x)
        x = self.out(x)
        return tNN.functional.softmax(x, dim=1)
    
class ModelCollection:
    def __init__(self, device, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10):
        self.device = device
        self.numOfModels = 9 # change 10 if use all models
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        self.model8 = model8
        self.model9 = model9
        self.model10 = model10
    def GetOutputFromModels(self, x):
        outputs = []
        #outputs.append(torch.softmax(self.model1(x), dim=1))
        outputs.append(torch.softmax(self.model2(x), dim=1))
        outputs.append(torch.softmax(self.model3(x), dim=1))
        outputs.append(torch.softmax(self.model4(x), dim=1))
        outputs.append(torch.softmax(self.model5(x), dim=1))
        outputs.append(torch.softmax(self.model6(x), dim=1))
        outputs.append(torch.softmax(self.model7(x), dim=1))
        outputs.append(torch.tensor(self.model8.Predict(self.device, self.model3, x), device=self.device).float())
        outputs.append(torch.tensor(self.model9.Predict(self.device, self.model2, x), device=self.device).float())
        outputs.append(torch.tensor(self.model10.Predict(self.device, self.model5, x), device=self.device).float())
        return torch.cat((outputs), dim=1)

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

def testingAll(device, model, modelCollection, dataLoaders, criterion):
    model.eval()
    testLoss = 0
    correct = 0
    classCorrect = defaultdict(int)
    classTotal = defaultdict(int)
    preds = []
    targets = []

    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(modelCollection.GetOutputFromModels(data))
            testLoss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=False)
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

            for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
                classTotal[t] += 1
                if t == p:
                    classCorrect[t] += 1

    overallAccuracy = accuracy_score(targets, preds)

    perClassAccuracy = {
        classID: 100.0 * classCorrect[classID] / classTotal[classID]
        for classID in sorted(classTotal.keys())
    }

    classReport = classification_report(targets, preds)

    print(f"\nTest set: Average loss: {testLoss / len(dataLoaders['test'].dataset):.4f}, "
          f"Overall Accuracy: {overallAccuracy:.2f}\n")

    for classID, acc in perClassAccuracy.items():
        print(f"Class {classID}: {acc:.2f}%")

    print("\nClassification Report:")
    print(classReport)

    return testLoss, overallAccuracy, perClassAccuracy, classReport

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
    
    model1Linear1 = torch.jit.load(f"../model1_Linear1/model1_Linear1_Scripted_{datasetName}.pt").to(device)
    model1Linear1.eval()
    
    model2CNN1 = torch.jit.load(f"../model2_CNN1/model2_CNN1_Scripted_{datasetName}.pt").to(device)
    model2CNN1.eval()
    
    model3CNN2 = torch.jit.load(f"../model3_CNN2/model3_CNN2_Scripted_{datasetName}.pt").to(device)
    model3CNN2.eval()
    
    model4CNN3 = torch.jit.load(f"../model4_CNN3/model4_CNN3_Scripted_{datasetName}.pt").to(device)
    model4CNN3.eval()
    
    model5CNN4 = torch.jit.load(f"../model5_CNN4/model5_CNN4_Scripted_{datasetName}.pt").to(device)
    model5CNN4.eval()
    
    model6CNN5 = torch.jit.load(f"../model6_CNN5/model6_CNN5_Scripted_{datasetName}.pt").to(device)
    model6CNN5.eval()
    
    model7CNN6 = torch.jit.load(f"../model7_CNN6/model7_CNN6_Scripted_{datasetName}.pt").to(device)
    model7CNN6.eval()
    
    model8SVM1 = pickle.load(open(f'../model8_SVM1_CNN2/model8_SVM1_{datasetName}.pkl', 'rb'))
    
    model9SVM2 = pickle.load(open(f'../model9_SVM2_CNN1/model9_SVM2_{datasetName}.pkl', 'rb'))
    
    model10SVM3 = pickle.load(open(f'../model10_SVM3_CNN4/model10_SVM3_{datasetName}.pkl', 'rb'))
    
    modelCollection = ModelCollection(
        device, 
        model1Linear1, 
        model2CNN1, 
        model3CNN2,
        model4CNN3,
        model5CNN4,
        model6CNN5,
        model7CNN6,
        model8SVM1,
        model9SVM2,
        model10SVM3
        )
    
    model = torch.jit.load(f'ensemble_model_stacking_Scripted_{datasetName}_noLinear.pt').to(device)
    criterion = tNN.CrossEntropyLoss().to(device)

    testingAll(device, model, modelCollection, dataLoaders, criterion)

if __name__ == '__main__':
    main()