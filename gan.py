import os
import torch

import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader, dataloader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm

img_save_path = 'images/gan'
os.makedirs(img_save_path, exist_ok=True)

epochs = 200
batch_size = 64
learning_rate = 0.0002
betas = (0.5, 0.999)
latent_dim = 100
img_size = 28
channels = 1
sample_interval = 400
img_shape = (channels, img_size, img_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor
img_save_interval = 400

dataset = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
)

data_loader = DataLoader(
    dataset = dataset,
    batch_size = batch_size,
    shuffle = True
)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
    

adversarial_loss = torch.nn.BCELoss().to(device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)

for epoch in range(epochs):
    with tqdm(data_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch}')
        for i, (img, _) in enumerate(t):
            # print(img.shape[0])
            valid = Variable(Tensor(img.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(img.size(0), 1).fill_(0.0), requires_grad=False)
            
            real_img = Variable(img.type(Tensor))
            
            # train generator
            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (img.shape[0], latent_dim))))
            gen_img = generator(z)
            g_loss = adversarial_loss(discriminator(gen_img), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # train discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_img), valid)
            fake_loss = adversarial_loss(discriminator(gen_img.detach()), fake)
            d_loss = (real_loss+fake_loss)/2
            d_loss.backward()
            optimizer_D.step()
            
            t.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())
            if i%img_save_interval == 0:
                save_image(gen_img[:25], f'{img_save_path}/epoch{epoch}_{i}.png', nrow=5, normalize=True)