import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


vgg_arch = {
    'VGG11' : [64, 'MP', 128, 'MP', 256, 256, 'MP', 512, 512, 'MP', 512, 512, 'MP'],
    'VGG13' : [64, 64, 'MP', 128, 128, 'MP', 256, 256, 'MP', 512, 512, 'MP', 512, 512, 'MP'],
    'VGG16' : [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP'],
    'VGG19' : [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512, 512, 'MP', 512, 512, 512, 512, 'MP']
}


img_channel = 3

class VGGNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=1000):
        super(VGGNet, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(vgg_arch['VGG11'])
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channel
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1),
                            # nn.BatchNorm2d(x),
                            nn.ReLU()
                            ]
                in_channels = x
            else:
                layers += [nn.MaxPool2d(2, 2)]
                
        return nn.Sequential(*layers)
    
net = VGGNet()

sample = torch.rand(64, 3, 224, 224)

print(net(sample).shape)