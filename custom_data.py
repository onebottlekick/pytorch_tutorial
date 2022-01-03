import os
import glob
import zipfile

from PIL import Image
from torch.utils.data import Dataset
from urllib import request


class HymenopteraDataset(Dataset):
    def __init__(self, root='data', download=True, train=True, transform=None):
        # self.file_list = file_list
        self.root = root
        self.download = download
        self.train = train
        self.transform = transform
        self.dataset_name = 'hymenoptera_data'
        self.dataset_path = os.path.join(self.root, self.dataset_name)
        
        if self.download:
            if not os.path.exists(self.dataset_path):
                url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
                zip_path = os.path.join(root, self.dataset_name+'.zip')
                request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path) as zip_:
                    zip_.extractall(self.root)
                os.remove(zip_path)
        
        if self.train:
            self.file_list = self.make_data_path_list()
        else:
            self.file_list = self.make_data_path_list()
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        label = [img_path[img_path.find(i): img_path.find(i)+4] for i in ['ants', 'bees'] if img_path.find(i) != -1][0]
        
        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1
        return img, label
    
    def make_data_path_list(self):        
        if self.train:
            target_path = os.path.join(self.dataset_path+'/train'+'/**/*.jpg')
        else:
            target_path = os.path.join(self.dataset_path+'/val'+'/**/*.jpg')

        path_list = [path for path in glob.glob(target_path)]

        return path_list
    

if __name__ == '__main__':
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms, models
    from tqdm import tqdm
    
    
    img_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    batch_size = 32

    epochs = 2
    learning_rate = 0.001
    momentum = 0.9
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader = DataLoader(
        dataset=HymenopteraDataset(
            root='data',
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=HymenopteraDataset(
            root='data',
            download=True,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    model.train()
    
    params_to_update = []
    update_param_names = ['classifier.6.weight', 'classifier.6.biaas']
    for name, param in model.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=params_to_update, lr=learning_rate, momentum=momentum)
    
    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader, unit='batch') as t:
            for data, target in t:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                t.set_description(f'Epoch {epoch+1}')
                t.set_postfix(train_loss={f'{loss.item():.4f}'})
        
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                _, preds = outputs.max(1)
                correct += (preds == target.data).sum()
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100*correct/len(val_loader.dataset)
        print(f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')