import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical




class Policy(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
        
        self.on_policy_reset()
        self.train()
        
    def on_policy_reset(self):
        self.log_probs = []
        self.rewards = []
        
    def forward(self, x):
        pdparam = self.model(x)
        return pdparam
        
    def act(self, state):
        x = torch.tensor(state.astype(np.float32))
        pdparam = self.forward(x)
        pd = Categorical(logits=pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()


def train(policy, optimizer, gamma=0.9):
    T = len(policy.rewards)
    returns = np.empty(T, dtype=np.float32)
    future_return = 0.0
    
    # LIFO(List)
    for t in reversed(range(T)):
        future_return = policy.rewards[t] + gamma*future_return
        returns[t] = future_return
    
    returns = torch.tensor(returns)
    log_probs = torch.stack(policy.log_probs)
    loss = torch.sum(-log_probs*returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    # gamma = 0.9
    episodes = 1000
    T = 200
    env = gym.make('CartPole-v0')
    in_features = env.observation_space.shape[0]
    out_features = env.action_space.n
    
    policy = Policy(in_features, out_features)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    for episode in range(episodes):
        state = env.reset()
        for t in range(T):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            env.render()
            if done:
                break
        
        loss = train(policy, optimizer)
        total_reward = sum(policy.rewards)
        solved = total_reward > 195
        policy.on_policy_reset()
        print(f'Episode {episode}, loss: {loss}, total_reward: {total_reward}, solved: {solved}')
        

if __name__ == '__main__':
    main()