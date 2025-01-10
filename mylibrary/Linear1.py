import torch.nn as tNN

class Linear1(tNN.Module):
    def __init__(self, inputChannel, inputSize, numClasses):
        super(Linear1, self).__init__()
        self.inputSize = inputSize * inputSize * inputChannel
        self.fc1 = tNN.Linear(self.inputSize,300)
        self.fc2 = tNN.Linear(300,300)
        self.out = tNN.Linear(300,numClasses)
    def forward(self, x):
        x = x.view(-1, self.inputSize)
        x = tNN.functional.relu(self.fc1(x))
        x = tNN.functional.relu(self.fc2(x))
        x = self.out(x)
        return tNN.functional.softmax(x, dim=1)