import torch
import torch.nn as tNN

class SVM3:
    def __init__(self, svmModel):
        self.SVMModel = svmModel
        
    def ExtractFeatures(self, device, cnnModel, dataLoader):
        features = []
        with torch.no_grad():
            for inputs in dataLoader:
                inputs = inputs.to(device)

                outputs = tNN.functional.relu(cnnModel.conv1(inputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = tNN.functional.relu(cnnModel.conv2(outputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = tNN.functional.relu(cnnModel.conv3(outputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = tNN.functional.relu(cnnModel.conv4(outputs))
                outputs = cnnModel.maxPool(outputs)
                outputs = outputs.view(-1, cnnModel.inputSize)
                
                batch_features = outputs.view(outputs.size(0), -1).cpu().numpy()
                features.extend(batch_features)

        return features
    
    def Predict(self, device, cnnModel, dataLoader):
        features = self.ExtractFeatures(device, cnnModel, dataLoader)
        return self.SVMModel.predict_proba(features)