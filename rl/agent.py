import os
import random
from collections import deque

import numpy as np
import torch

from env import BLOCK_SIZE, ENV, Direction, Point
from model import LinearQNet
from utils import QTrainer, plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_simulation = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet()
        self.trainer = QTrainer(self.model, LR, self.gamma)
    
    def get_state(self, env):
        head = env.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = env.direction == Direction.LEFT
        dir_r = env.direction == Direction.RIGHT
        dir_u = env.direction == Direction.UP
        dir_d = env.direction == Direction.DOWN

        state = [
            (dir_r and env.is_collision(point_r)) or 
            (dir_l and env.is_collision(point_l)) or 
            (dir_u and env.is_collision(point_u)) or 
            (dir_d and env.is_collision(point_d)),

            (dir_u and env.is_collision(point_r)) or 
            (dir_d and env.is_collision(point_l)) or 
            (dir_l and env.is_collision(point_u)) or 
            (dir_r and env.is_collision(point_d)),

            (dir_d and env.is_collision(point_r)) or 
            (dir_u and env.is_collision(point_l)) or 
            (dir_r and env.is_collision(point_u)) or 
            (dir_l and env.is_collision(point_d)),
            
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            env.food.x < env.head.x,
            env.food.x > env.head.x,
            env.food.y < env.head.y,
            env.food.y > env.head.y
            ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            samples = random.sample(self.memory, BATCH_SIZE)
        else:
            samples = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*samples)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_simulation
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1
            
        return action
    

def train():
    scores = []
    avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    if os.path.exists(agent.model.model_path):
        agent.model.load(agent.model.model_path)
    env = ENV()
    while True:
        state_old = agent.get_state(env)
        action = agent.get_action(state_old)
        reward, game_over, score = env.play_step(action)
        state_new = agent.get_state(env)
        
        agent.train_short_memory(state_old, action, reward, state_new, game_over)
        
        agent.remember(state_old, action, reward, state_new, game_over)
        
        if game_over:
            env.reset()
            agent.n_simulation += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print('Simulation', agent.n_simulation, 'Score', score, 'Record', record)
            
            scores.append(score)
            total_score += score
            avg_score = total_score/agent.n_simulation
            avg_scores.append(avg_score)
            
            plot(scores, avg_scores)
                
if __name__ == '__main__':
    train()
