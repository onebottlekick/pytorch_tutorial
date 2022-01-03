import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision import models
from torchvision.utils import save_image

model = models.vgg19(pretrained=True).features
# print(model)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image.to(device)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_size = 356

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[], std=[])
])

img_path = 'images/style_transfer'
target = 'target.png'
target_path = os.path.join(img_path, target)

style = 'style.png'
style_path = os.path.join(img_path, style)

original_img = load_image(target_path)
style_img = load_image(style_path)

model = VGG().to(device).eval()

# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated = original_img.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 0.001
alpha = 0.97 # target factor
beta = 0.03 # style factor

optimizer = torch.optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_img_features = model(style_img)

    style_loss = 0
    original_loss = 0
    
    for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_img_features):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2) 
        
        G = gen_feature.view(channel, height*width).mm(gen_feature.view(channel, height*width).t())
        A = style_feature.view(channel, height*width).mm(style_feature.view(channel, height*width).t())

        style_loss += torch.mean((G - A)**2)

    total_loss = alpha*original_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 200 == 0:
        print(total_loss)
        save_image(generated, os.path.join(img_path, f'gen_{step}_{target}'))