import torch.nn as tNN

class CNN3(tNN.Module):
    def __init__(self, inputChannel, inputSize, numClassess):
        super(CNN3, self).__init__()
        self.inputSize = inputSize
        self.outputChannel = 256
        self.conv1 = tNN.Conv2d(inputChannel, 16, kernel_size=3)
        self.conv2 = tNN.Conv2d(16, 32, kernel_size=3)
        self.conv3 = tNN.Conv2d(32, 64, kernel_size=3)
        self.conv4 = tNN.Conv2d(64, 128, kernel_size=3)
        self.conv5 = tNN.Conv2d(128, self.outputChannel, kernel_size=3)
        self.convDrop = tNN.Dropout2d()
        self.fcDrop = tNN.Dropout(0.5)
        self.maxPool = tNN.MaxPool2d(2, 2)
        self.inputSize = ((self.inputSize - 3 + 2*0)//1) + 1 # 3x3 conv stride 1 zero padding
        self.inputSize = ((self.inputSize - 3 + 2*0)//1) + 1 # 3x3 conv stride 1 zero padding
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = ((self.inputSize - 3 + 2*0)//1) + 1 # 3x3 conv stride 1 zero padding
        self.inputSize = ((self.inputSize - 3 + 2*0)//1) + 1 # 3x3 conv stride 1 zero padding
        self.inputSize = ((self.inputSize - 3 + 2*0)//1) + 1 # 3x3 conv stride 1 zero padding
        self.inputSize = ((self.inputSize - 2 + 2*0)//2) + 1 # maxpool 2x2 stride 2
        self.inputSize = int(self.inputSize * self.inputSize * self.outputChannel)
        self.fc1 = tNN.Linear(self.inputSize, 512)
        self.fc2 = tNN.Linear(512, 128)
        self.fc3 = tNN.Linear(128, numClassess)
   
    def forward(self, x):
        x = tNN.functional.relu(self.conv1(x))
        x = tNN.functional.relu(self.conv2(x))
        x = self.maxPool(x)
        x = tNN.functional.relu(self.conv3(x))
        x = tNN.functional.relu(self.conv4(x))
        x = tNN.functional.relu(self.conv5(x))
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