import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, dataloader

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

batch_size = 32
epochs = 10
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = datasets.CIFAR10(
    root = 'data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_data = datasets.CIFAR10(
    root = 'data',
    download = True,
    transform = transforms.ToTensor(),
    train = False
)

train_loader = DataLoader(
    dataset = train_data,
    batch_size = batch_size,
    shuffle = True
)

test_loader = DataLoader(
    dataset = test_data,
    batch_size = batch_size,
    shuffle = True
)

# torch.Size([3, 32, 32])
# print(test_loader.dataset[0][0].shape)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # nn.BatchNorm2d(),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32*6*6, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        return self.model(x)
    

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


history = {
    "train_loss_list" : [],
    "test_loss_list" : [],
    "train_acc_list" : [],
    "test_acc_list" : []
    }

for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0
    correct = 0
    with tqdm(train_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch}')
        for image, label in t:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            train_loss += criterion(output, label).item()
            pred = output.max(1, keepdims=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100*correct/len(train_loader.dataset)
    
    history["train_loss_list"].append(train_loss)
    history["train_acc_list"].append(train_accuracy)
            
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
    
    history['test_loss_list'].append(test_loss)
    history['test_acc_list'].append(test_accuracy)
    

def plot_history(history, save=False):
    epochs = len(history['train_loss_list'])
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('EPOCH')
    plt.ylabel('Loss')
    plt.plot(history['train_loss_list'], 'red', label='train')
    plt.plot(history['test_loss_list'], 'blue', label='test')
    plt.legend()
    plt.xticks(np.arange(0, epochs, 1))
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(history['train_acc_list'], 'red', label='train')
    plt.plot(history['test_acc_list'], 'blue', label='test')
    plt.grid(True)
    plt.legend()
    plt.xticks(np.arange(0, epochs, 1))
    if save:
        import os
        import datetime
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/cifar10_{str(datetime.datetime.now())[:10]}.png')
    else:
        plt.show()
    
plot_history(history)