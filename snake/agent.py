import torch
import random
import numpy as np
from game import SnakeGameAI
from collections import deque
from game import Point, SnakeGameAI, Direction
import matplotlib.pyplot as plt
from model import QTrainer, Linear_QNet
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.005

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0     # epsilon greedy
        self.gamma = 0.9     # discount factor
        self.memory = deque(maxlen=MAX_MEMORY)       # if > then popleft() the old one bye
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    
    def get_state(self, game):

        '''state = [Danger straight, Danger right, Danger left,
                 Left, Right, Up, Down,
                  food left, food right, food up, food down]'''

        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_short_memory(self, state, action, reward, next_action, done):
        self.trainer.train_step(state, action, reward, next_action, done)


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE :
            # list of tuples
            sample = random.sample(self.memory, BATCH_SIZE)
        else :
            sample = self.memory
        
        states, actions, rewards, next_actions, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_actions, dones)



    def get_action(self, state):
        # we want less exploration in a trained model
        self.epsilon = max(0.05, 1.0 - self.n_games / 1000) 
        # [straight, right, left]
        final_move = [0,0,0]
        if random.random() < self.epsilon :
            # explore
            move = random.randint(0,2)
        else : 
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            
        final_move[move] = 1
        return final_move
    

def train():
    plot_scores, plot_mean_scores = [], []
    total_score, record = 0, 0
    agent = Agent()
    game = SnakeGameAI()

    while True :
        # get old state 
        old_state = agent.get_state(game)

        # get move 
        action =  agent.get_action(old_state)

        # play => get reward => get new state
        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, action, reward, new_state, done)

        # remember this case
        agent.remember(old_state, action, reward, new_state, done)

        if done :
            # train long memory and plot the result

            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record :
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__" : 
    train()

