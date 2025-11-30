import random
import numpy as np
import torch
from utils.mcts import MCTS
from tqdm.notebook import trange
import torch.nn.functional as F

class AlphaZero:

    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game,args,model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state,player)
            actions_probs = self.mcts.search(neutral_state)
            memory.append((neutral_state,actions_probs,player))

            temperature_actions_probs = actions_probs ** (1/self.args['temperature']) 
            valid_moves = self.game.get_valid_moves(state)
            temperature_actions_probs *= valid_moves
            temperature_actions_probs /= np.sum(temperature_actions_probs)


            action = np.random.choice(self.game.action_size,p=temperature_actions_probs)
            state = self.game.get_next_state(state,action,player)
            value, is_terminal = self.game.get_value_and_terminated(state,action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state,hist_actions_probs,hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_actions_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)
    

    def train(self,memory):
        random.shuffle(memory)
        for batchIdx in range(0,len(memory),self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory)-1,batchIdx+self.args['batch_size'])]
            state,policy_targets,value_targets = zip(*sample)

            state,policy_targets,value_targets = np.array(state),np.array(policy_targets),np.array(value_targets).reshape(-1,1)

            state = torch.tensor(state,dtype=torch.float32,device=self.model.device)
            policy_targets = torch.tensor(policy_targets,dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets,dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy,policy_targets)
            value_loss = F.mse_loss(out_value,value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
        
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"trained_models/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(),f"optimizer_{iteration}.pt")
