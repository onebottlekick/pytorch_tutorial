import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, img_channel=1):
        super(LeNet, self).__init__()
        self.img_channel = img_channel
        
        self.conv1 = nn.Conv2d(self.img_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, x):
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
        
model = LeNet()

a = torch.rand(64, 1, 32, 32)
print(model(a).shape)