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

model = torch.jit.load("model2_CNN1_Scripted.pt")
model.eval()

data, target = mnistTestData[238]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
data = data.unsqueeze(0).to(device)

output = model(data)

prediction = output.argmax(dim = 1, keepdim = True).item()

print(f"Prediction: {prediction}")

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