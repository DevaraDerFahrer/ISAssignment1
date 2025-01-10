import torch
import torch.nn as tNN
import time
import torch.optim as tOptim
import pickle
import mylibrary

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
        for data, target in dataLoaders['validation']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            testLoss += criterion(output, target).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    testLoss /= len(dataLoaders['validation'].dataset)
    testAccuracy = correct/len(dataLoaders['validation'].dataset)
    print(f"\nTest set: Average loss: {testLoss:.4f}, Accuracy {correct}/{len(dataLoaders['validation'].dataset)} ({100. * correct / len(dataLoaders['validation'].dataset):.0f}%)\n")
    return testLoss, testAccuracy

def main():
    programStartTime = time.time()

    datasetName = input("Choose dataset[mnist/cifar10]: ")
    while not(datasetName == "mnist" or datasetName == "cifar10" or datasetName == "exit"):
        datasetName = input("Choose dataset[mnist/cifar10]: ")
    
    inputSize = 32
    inputChannel = 0
    numClassess = 10
    
    if datasetName == "mnist":
        inputChannel = 1
    else:
        inputChannel = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mylibrary.CNN3(inputChannel, inputSize, numClassess).to(device)
    criterion = tNN.CrossEntropyLoss().to(device)
    optimizer = tOptim.Adam(model.parameters(),lr=0.001)
    
    dataLoaders = pickle.load(open(f'../data/dataLoaders{datasetName}.pkl', 'rb'))
    
    numOfEpoch = 100
    
    losses = []
    accuracies = []
    
    bestAccuracy = 0.0
    for epoch in range(1, numOfEpoch+1):
        training(device, model, dataLoaders, criterion, optimizer, epoch)
        testLoss, testAccuracy = testing(device, model, dataLoaders, criterion)
        losses.append(testLoss)
        accuracies.append(testAccuracy)
        
        if testAccuracy >= bestAccuracy:
            bestAccuracy = testAccuracy
            torch.save(model, f'model4_CNN3_{datasetName}.pt')
            modelScripted = torch.jit.script(model)
            modelScripted.save(f'model4_CNN3_Scripted_{datasetName}.pt')
    
    mylibrary.SavePlotAsVectors(
        x=range(1, numOfEpoch+1), y=losses,
        title="Training Loss Over Epochs",
        xlabel="Epochs", ylabel="Loss",
        filename=f"training_loss_{datasetName}",
        output_dir="results"
    )
    pickle.dump(losses, open(f'results/losses.pkl', 'wb'))

    mylibrary.SavePlotAsVectors(
        x=range(1, numOfEpoch+1), y=accuracies,
        title="Test Accuracy Over Epochs",
        xlabel="Epochs", ylabel="Accuracy (%)",
        filename=f"test_accuracy_{datasetName}",
        output_dir="results"
    )
    pickle.dump(accuracies, open(f'results/accuracies.pkl', 'wb'))
    
    print(f"Training done, elapsed time: {time.time() - programStartTime}")

if __name__ == '__main__':
    main()