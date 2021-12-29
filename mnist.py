import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm


epochs = 10
batch_size = 32
learning_rate = 0.001
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_interval = 200

train_data = MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_data = MNIST(
    root = './data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        # self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(32*5*5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        # self.pool3 = nn.MaxPool1d(2, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = x.view(-1, 5*5*32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
model = CNN().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# print(model)
for epoch in range(1, epochs+1):
    # train
    model.train()
    with tqdm.tqdm(train_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch:3d}')
        for batch_idx, (image, label) in enumerate(t):
            image = image.to(device)
            # print(image.shape)
            label = label.to(device)
            # print(label.shape)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=f'{loss.item():.4f}')
            # if batch_idx % log_interval == 0:
            #     print(f'[train epoch: {epoch}/{epochs} ({100*batch_idx/len(train_loader):.2f})%] [train loss = {loss.item():.4f}]')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    test_accuracy = 100*correct/len(test_loader.dataset)
    print(f'val_accuracy={test_accuracy:.2f}, val_loss={test_loss:.4f}')
