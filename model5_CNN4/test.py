import torch
import pickle
import mylibrary
from collections import defaultdict

def main():
    datasetName = input("Choose dataset[mnist/cifar10]: ")
    while not (datasetName == "mnist" or datasetName == "cifar10" or datasetName == "exit"):
        datasetName = input("Choose dataset[mnist/cifar10]: ")

    if datasetName == "exit":
        return

    dataLoaders = pickle.load(open(f'../data/dataLoaders{datasetName}.pkl', 'rb'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(f"../model5_CNN4/model5_CNN4_Scripted_{datasetName}.pt").to(device)
    model.eval()

    correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for data, target in dataLoaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)

            # Update overall correct count
            correct += pred.eq(target).sum().item()

            # Update counts for each class
            for t, p in zip(target, pred):
                if t == p:
                    class_correct[t.item()] += 1
                class_total[t.item()] += 1

    # Calculate overall accuracy
    overall_accuracy = correct / len(dataLoaders['test'].dataset)
    print(f"Overall Accuracy: {overall_accuracy:.2f} ({100. * overall_accuracy:.2f}%)")

    # Calculate and print accuracy for each class
    print("Accuracy for individual classes:")
    for class_id in sorted(class_total.keys()):
        class_accuracy = 100.0 * class_correct[class_id] / class_total[class_id]
        print(f"Class {class_id}: {class_accuracy:.2f}%")

    return

if __name__ == '__main__':
    main()