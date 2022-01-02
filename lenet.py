import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


img_channel = 1
# img_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
epochs = 10

train_loader = DataLoader(
    dataset=datasets.MNIST(
        root='data',
        train=True,
        transform=transforms.Compose([
            # transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    ),
    shuffle=True,
    batch_size=batch_size
)

test_loader = DataLoader(
    dataset=datasets.MNIST(
        root='data',
        train=False,
        transform=transforms.Compose([
            # transforms.Resize(img_size),
            transforms.ToTensor()
        ])
    ),
    shuffle=True,
    batch_size=batch_size
)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(img_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = nn.ZeroPad2d(2)(x) # make image 28x28 -> 32x32
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
        
model = LeNet().to(device)

# a = torch.rand(64, 1, 32, 32).to(device)
# b = model(a)
# print(b.shape)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    t = tqdm(train_loader, unit='batch')
    model.train()
    for data, target in t:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        t.set_description(f'Epoch {epoch}')
        t.set_postfix(loss=f'{loss.item():.4f}')
        
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            test_loss += criterion(pred, target).item()
            pred = pred.max(1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100*correct/len(test_loader.dataset)
    print(f'val_loss={test_loss:.4f}, val_accuracy={test_accuracy:.2f}')