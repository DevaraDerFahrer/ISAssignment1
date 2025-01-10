import torch.nn as tNN

class CNN5(tNN.Module):
    def __init__(self, inputChannel, inputSize, numClassess):
        super(CNN5, self).__init__()
        self.outputChannel = 128
        self.conv1 = tNN.Conv2d(inputChannel, 128, kernel_size=5, stride=2, padding=2)
        self.conv2 = tNN.Conv2d(128, 128, kernel_size=3, padding="same")
        self.conv3 = tNN.Conv2d(128, 128, kernel_size=3, padding="same")
        self.conv4 = tNN.Conv2d(128, self.outputChannel, kernel_size=3, padding="same")
        self.convDrop = tNN.Dropout2d()
        self.fcDrop = tNN.Dropout(0.5)
        self.maxPool = tNN.MaxPool2d(2, 2)
        self.inputSize = ((inputSize - 5 + 2*2)//2) + 1 # 5x5 conv stride 2
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = int(self.inputSize * self.inputSize * self.outputChannel)
        self.fc1 = tNN.Linear(self.inputSize, 128)
        self.fc2 = tNN.Linear(128, 128)
        self.fc3 = tNN.Linear(128, numClassess)
   
    def forward(self, x):
        x = tNN.functional.relu(self.conv1(x))
        x = self.maxPool(x)
        x = tNN.functional.relu(self.conv2(x))
        x = self.maxPool(x)
        x = tNN.functional.relu(self.conv3(x))
        x = self.maxPool(x)
        x = tNN.functional.relu(self.conv4(x))
        x = self.maxPool(x)
        x = x.view(-1, self.inputSize)
        x = self.fc1(x)
        x = self.fcDrop(x)
        x = tNN.functional.relu(x)
        x = self.fc2(x)
        x = self.fcDrop(x)
        x = tNN.functional.relu(x)
        x = self.fc3(x)
        return x