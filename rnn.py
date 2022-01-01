import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
epochs = 2

train_dataset = datasets.MNIST(
    root='data',
    download=True,
    transform=transforms.ToTensor(),
    train=True
)

test_dataset = datasets.MNIST(
    root='data',
    download=True,
    transform=transforms.ToTensor(),
    train=False
)

train_loader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=batch_size
)

test_loader = DataLoader(
    dataset=test_dataset,
    shuffle=True,
    batch_size=batch_size
)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    with tqdm(train_loader, unit='batch') as t:
        t.set_description(f'Epoch {epoch}')
        for data, target in t:
            model.train()
            data = data.to(device).squeeze(1)
            target = target.to(device)

            
            pred = model(data)
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t.set_postfix(loss=f'{loss.item():.4f}')
            
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():                
        for data, target in test_loader:
            data = data.to(device).squeeze(1)
            target = target.to(device)
            pred = model(data)
            test_loss += criterion(pred, target).item()
            _, pred = pred.max(1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100*correct/len(test_loader.dataset)
    print(f'val_loss={test_loss:.4f}, val_acc={test_accuracy:.2f}')
    
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('checking accuracy on training datda')
    else:
        print('checking accuracy on test data')

    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            pred = model(x)
            _, pred = pred.max(1)
            num_correct += (pred==y).sum()
            num_samples += pred.size(0)

        print(f'accuracy={num_correct/num_samples:.2f}')
    model.train()
    
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)