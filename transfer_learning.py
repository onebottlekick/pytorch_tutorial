import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
epochs = 10

# data_transforms = {
#     'train' : transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ]),
#     'val' : transforms.Compose([
#         transforms.CenterCrop(224),
#         transforms.Resize(256),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
# }

# image_datasets = {x : datasets.ImageFolder('data/hymenoptera_data', data_transforms[x]) for x in ['train', 'val']}
train_data = datasets.ImageFolder(
    root = 'data/hymenoptera_data/train',
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
)

test_data = datasets.ImageFolder(
    root = 'data/hymenoptera_data/val',
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
)

# dataloaders = {x : torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size = batch_size,
    shuffle = True
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_data,
    batch_size = batch_size,
    shuffle = True
)


model = models.resnet18(pretrained=False).to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, epochs+1):
    model.train()
    with tqdm(train_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch}')
        for image, label in t:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=f'{loss.item():.4f}')
            
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()
            pred = output.max(1, keepdims=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100*correct/len(test_loader.dataset)
    print(f'val_loss={test_loss:.4f}, val_accuracy={test_accuracy:.2f}')