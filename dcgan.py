import torch
import os
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm


img_size = 32
img_channel = 1
img_shape = (img_channel, img_size, img_size)
batch_size = 64
latent_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.0002
betas = (0.5, 0.999)
epochs = 200
FloatTensor = torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor
img_save_interval = 400

img_save_path = 'images/dcgan'
os.makedirs(img_save_path, exist_ok=True)


dataset = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
)

data_loader = DataLoader(
    dataset = dataset,
    shuffle=True,
    batch_size=batch_size
)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128*self.init_size**2))
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channel, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.model(out)
        return img
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def block(in_filters, out_filters, normalize=True):
            layer = [
                nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
                ]
            if normalize:
                layer.append(nn.BatchNorm2d(out_filters, 0.8))
            return layer
        
        self.model = nn.Sequential(
            *block(img_channel, 16, normalize=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128)
        )
        
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128*ds_size**2, 1),
            nn.Sigmoid()
            )
        
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
    

adversarial_criterion = nn.BCELoss().to(device)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)

for epoch in range(epochs):
    with tqdm(data_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch}')
        for i, (img, _) in enumerate(t):
            # print(img.shape[0])
            valid = torch.ones(img.size(0), 1, requires_grad=False).type(FloatTensor).to(device)
            fake = torch.zeros(img.size(0), 1, requires_grad=False).type(FloatTensor).to(device)
            
            real_img = img.type(FloatTensor).to(device)
            
            # train generator
            optimizer_G.zero_grad()
            z = torch.rand(img.shape[0], latent_dim).to(device)
            gen_img = generator(z)
            g_loss = adversarial_criterion(discriminator(gen_img), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # train discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_criterion(discriminator(real_img), valid)
            fake_loss = adversarial_criterion(discriminator(gen_img.detach()), fake)
            d_loss = (real_loss+fake_loss)/2
            d_loss.backward()
            optimizer_D.step()
            
            t.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())
            if i%img_save_interval == 0:
                save_image(gen_img[:25], f'{img_save_path}/epoch{epoch}_{i}.png', nrow=5, normalize=True)
            