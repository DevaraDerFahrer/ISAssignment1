import PIL.Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Grayscale
from torchvision import datasets
import torch.nn as tNN
import PIL
import matplotlib.pyplot as plt

testData = datasets.CIFAR10(
    root="../data",
    train= False,
    transform=transforms.Compose([ToTensor(),Grayscale(num_output_channels=1)]),
    download=True
)

datasetName = "cifar10"

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = torch.jit.load(f"model2_CNN1_Scripted_{datasetName}.pt")
model.eval()

data, target = testData[123]

print(target)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
data = data.unsqueeze(0).to(device)

output = model(data)

prediction = output.argmax(dim = 1, keepdim = True).item()

print(f"Prediction: {classes[prediction]} Actual value: {classes[target]}")

image = data.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap="gray")
plt.show()

pic = PIL.Image.open("../data/manualtestdata/2D0.png").convert("L")
transformF = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
picTensor = transformF(pic).to(device)

transformF = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),torchvision.transforms.PILToTensor()])

output = model(picTensor)

prediction = output.argmax(dim = 1, keepdim = True).item()

print(f"Prediction: {prediction}")

image = picTensor.squeeze(0).squeeze(0).cpu().numpy()

plt.imshow(image, cmap="gray")
plt.show()