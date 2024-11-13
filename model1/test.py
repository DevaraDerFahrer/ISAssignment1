import PIL.Image
import torch
import torchvision 
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch.nn as tNN
import PIL

mnistTestData = datasets.MNIST(
    root="../data",
    train= False,
    transform=ToTensor(),
    download=True
)

class model1Linear(tNN.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = tNN.Linear(784,36)
        self.fc2 = tNN.Linear(36,36)
        self.out = tNN.Linear(36,10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = tNN.functional.relu(self.fc1(x))
        x = tNN.functional.relu(self.fc2(x))
        x = self.out(x)
        return tNN.functional.softmax(x)


model = torch.load("trainedModel.pt")
model.eval()

data, target = mnistTestData[305]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.unsqueeze(0).to(device)

output = model(data)

prediction = output.argmax(dim = 1, keepdim = True).item()

print(f"Prediction: {prediction}")

image = data.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap="gray")
plt.show()

pic = PIL.Image.open("../manualtestdata/9D0.png").convert("L")
transformF = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
picTensor = transformF(pic)

transformF = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),torchvision.transforms.PILToTensor()])

output = model(picTensor)

prediction = output.argmax(dim = 1, keepdim = True).item()

print(f"Prediction: {prediction}")

image = picTensor.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap="gray")
plt.show()