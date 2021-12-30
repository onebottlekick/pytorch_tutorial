import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms, datasets

from tqdm import tqdm
import os

os.makedirs('plots/ae', exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 32
epochs = 10

train_data = datasets.FashionMNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
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

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28)
        )
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    
model = AE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(1, epochs+1):
    model.train()
    with tqdm(train_loader) as t:
        t.set_description(f'Epoch {epoch}')
        for image, _ in t:
            image = image.view(-1, 28*28).to(device)
            target = image.view(-1, 28*28).to(device)
            optimizer.zero_grad()
            encoded, decoded = model.forward(image)
            loss = criterion(decoded, target)
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=f'{loss.item():.4f}')
            
    model.eval()
    test_loss = 0
    real_image = []
    gen_image = []
    
    with torch.no_grad():
        for image, _ in test_loader:
            image = image.view(-1, 28*28).to(device)
            target = image.view(-1, 28*28).to(device)
            encoded, decoded = model.forward(image)
            
            test_loss += criterion(decoded, image).item()
            real_image.append(image.to('cpu'))
            gen_image.append(decoded.to('cpu'))
    test_loss /= len(test_loader.dataset)
    print(f'val_loss={test_loss:.4f}')
    

    for i in range(10):
        img = np.reshape(real_image[0][i], (28, 28))
        plt.subplot(2, 10, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        img = np.reshape(gen_image[0][i], (28, 28))
        plt.subplot(2, 10, i+11)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.savefig(f'plots/ae/epoch{epoch}.png')
    
# a = torch.randn(32).to('cuda')
# b = model.decoder(a).to('cpu').detach().numpy()
# plt.imshow(b.reshape(28, 28))
# plt.show()