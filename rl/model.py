import os

import torch
import torch.nn as nn


class LinearQNet(nn.Module):
    def __init__(self, in_features=11, hidden_size=256, out_features=3, root='model', model_name='model.pt'):
        super(LinearQNet, self).__init__()
        self.root = root
        self.model_name = model_name
        self.model_path = os.path.join(self.root, self.model_name)
        
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)
        
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def save(self):
        os.makedirs(self.root, exist_ok=True)
        torch.save(self.state_dict(), self.model_path)
        
    def load(self, model='model.pt'):
        self.load_state_dict(torch.load(self.model_path))
