import os
import torch

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


img_size = 28
img_channel = 1
img_shape = (img_channel, img_size, img_size)
batch_size = 512
latent_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.0002
betas = (0.5, 0.999)
epochs = 200
img_save = False
img_save_path = 'images/gan'
log_interval = 400

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

writer_real = SummaryWriter('logs/mnist_gan/real')
writer_fake = SummaryWriter('logs/mnist_gan/fake')
step = 0

for epoch in range(epochs):
    with tqdm(data_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch}')
        for i, (img, _) in enumerate(t):
            # print(img.shape[0])
            valid = torch.ones(img.size(0), 1, device=device, dtype=torch.float32)
            fake = torch.zeros(img.size(0), 1, device=device, dtype=torch.float32)
            
            real_img = img.to(device)
            
            # train generator
            optimizer_G.zero_grad()
            z = torch.rand(img.size(0), latent_dim, device=device)
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
            if img_save:
                os.makedirs(img_save_path, exist_ok=True)
                if i%log_interval == 0:
                    save_image(gen_img[:25], f'{img_save_path}/epoch{epoch}_{i}.png', nrow=5, normalize=True)
            
            if i % log_interval == 0:
                with torch.no_grad():
                    img_grid_real = make_grid(real_img[:32], normalize=True)
                    img_grid_fake = make_grid(gen_img[:32], normalize=True)

                    writer_real.add_image('Real', img_grid_real, global_step=step)
                    writer_fake.add_image('Fake', img_grid_fake, global_step=step)
                step += 1