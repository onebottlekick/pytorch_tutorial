import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython import display


def plot(scores, avg_scores):
    plt.ion()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('training...')
    plt.xlabel('Number of simulations')
    plt.ylabel('Score')
    plt.plot(scores, color='blue', label='score')
    plt.plot(avg_scores, color='red', label='avg_score')
    plt.ylim(ymin=0)
    plt.legend(loc='upper left')
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(avg_scores) -1, avg_scores[-1], str(avg_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            game_over = (game_over, )
            
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma*torch.max(self.model(next_state[idx]))
                
            target[idx][action.argmax().item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
