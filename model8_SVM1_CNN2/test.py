import torch
import pickle
import torch.nn as tNN
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from collections import defaultdict

def testing(device, modelCollection, dataLoaders):
    outputTotal = []
    targetTotal = []
    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = modelCollection.Vote(data, len(data))
            outputTotal.extend(output)
            targetTotal.extend(target.cpu().numpy())
    
    print("Overall Accuracy:", accuracy_score(targetTotal, outputTotal))
    print("Classification Report:")
    print(classification_report(targetTotal, outputTotal))
    return 

def main():
    datasetName = input("Choose dataset[mnist/cifar10]: ")
    while not (datasetName == "mnist" or datasetName == "cifar10" or datasetName == "exit"):
        datasetName = input("Choose dataset[mnist/cifar10]: ")

    if datasetName == "exit":
        return
    elif datasetName == "mnist":
        validationLabels = pickle.load(open('testLabels_mnist.pkl', 'rb'))
    elif datasetName == "cifar10":
        validationLabels = pickle.load(open('testLabels_cifar10.pkl', 'rb'))
    else:
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnnModel = torch.jit.load(f"../model3_CNN2/model3_CNN2_Scripted_{datasetName}.pt").to(device)
    cnnModel.eval()

    modelSVM = pickle.load(open(f'model8_SVM1_{datasetName}.pkl', 'rb'))

    dataLoaders = pickle.load(open(f'../data/dataLoaders{datasetName}.pkl', 'rb'))

    outputTotal = []
    targetTotal = []

    # Class-level accuracy tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = modelSVM.Predict(device, cnnModel, data)
            predictions = np.argmax(output, axis=1)
            outputTotal.extend(predictions)
            targetTotal.extend(target.cpu().numpy())

            # Update per-class metrics
            for t, p in zip(target.cpu().numpy(), predictions):
                if t == p:
                    class_correct[t] += 1
                class_total[t] += 1

    # Overall accuracy
    overall_accuracy = accuracy_score(targetTotal, outputTotal)
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

    # Per-class accuracy
    print("Per-Class Accuracy:")
    for class_id in sorted(class_total.keys()):
        class_accuracy = 100.0 * class_correct[class_id] / class_total[class_id]
        print(f"Class {class_id}: {class_accuracy:.2f}%")

    # Classification report
    print("Classification Report:")
    print(classification_report(targetTotal, outputTotal))

    return 

if __name__ == '__main__':
    main()