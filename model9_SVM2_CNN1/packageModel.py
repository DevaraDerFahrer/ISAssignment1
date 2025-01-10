import torch
import pickle
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import mylibrary
import mylibrary.SVM1

def main():
    
    programStartTime = time.time()

    datasetName = input("Choose dataset[mnist/cifar10]: ")
    while not(datasetName == "mnist" or datasetName == "cifar10" or datasetName == "exit"):
        datasetName = input("Choose dataset[mnist/cifar10]: ")
    
    model = pickle.load(open(f'../model9_SVM2_CNN1/model9_SVM2_unpackaged_{datasetName}.pkl', 'rb'))
    packagedModel = mylibrary.SVM2(model)
    pickle.dump(packagedModel, open(f'model9_SVM2_{datasetName}.pkl', 'wb'))
    print(f"model saved, elapsed time: {time.time() - programStartTime}")

if __name__ == '__main__':
    main()