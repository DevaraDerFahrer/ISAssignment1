import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
data_dir = "./data"  # Replace with your dataset path

# Data transformation and loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(
            root="../data",
            train= True,
            transform=transform,
            download=True
        )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Split data into train and validation
train_data, val_data = train_test_split(train_dataset, test_size=0.2, random_state=42)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Pre-trained model for feature extraction
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()

# Remove the final layer to use as a feature extractor
model = torch.nn.Sequential(*(list(model.children())[:-1]))

def extract_features(loader, model):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Extract features for training and validation sets
train_features, train_labels = extract_features(train_loader, model)
val_features, val_labels = extract_features(val_loader, model)

# Train SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(train_features, train_labels)

# Validate SVM classifier
val_predictions = svm.predict(val_features)
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {accuracy:.4f}")
print(classification_report(val_labels, val_predictions, target_names=train_dataset.classes))
