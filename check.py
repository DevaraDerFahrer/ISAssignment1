import pickle

validationFeaturesSVM2 = pickle.load(open(f'model9_SVM2_CNN3/trainFeatures_cifar10.pkl', 'rb'))
validationFeaturesSVM3 = pickle.load(open(f'model10_SVM3_CNN4/trainFeatures_cifar10.pkl', 'rb'))

print(len(validationFeaturesSVM2[0]))
print(len(validationFeaturesSVM3[0]))