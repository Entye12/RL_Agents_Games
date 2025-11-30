import numpy as np
import math
import torch

class Node:

    def __init__(self,game,args,state,parent=None,action_taken=None,prior=0,visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.visit_count = visit_count
        self.children = []
        self.value_sum = 0

    # Is the node expanded ?
    def is_fully_expended(self):
        return len(self.children)>0
    
    # Select the child with the highest UCB score
    def select(self):
        best_child, best_ucb = None, - math.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb :
                best_child, best_ucb = child, ucb
        return best_child
    
    # Get the UCB score : UCB = Q_value + exploration term
    def get_ucb(self,child):
        if child.visit_count == 0 :
            Q_value = 0
        else :
            Q_value = 1 - ((child.value_sum /child.visit_count) + 1)/2
        ucb = Q_value + self.args['C']*(math.sqrt(self.visit_count)/(child.visit_count + 1))*child.prior
        return ucb
    
    # Using a policy PI to expand our node
    def expand(self,policy):
        for action,prob in enumerate(policy):
            if prob > 0 : 
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state,action,player=1)
                child_state = self.game.change_perspective(child_state,player=-1)

                child = Node(self.game,self.args,child_state,parent=self,action_taken=action,prior=prob)
                self.children.append(child)

    # Backprogate the value at the end 
    def backpropagate(self,value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent != None :
            value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value)

        
""" MCTS """

class MCTS:

    def __init__(self,game,args,model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self,state):

        # 1. Expanding the root
        root = Node(self.game,self.args,self.state,parent=None,action_taken=None,prior=0,visit_count=1)
        #    using a neural network to generate the policy we are expanding : get the logits
        policy,v = self.model(
            torch.tensor(self.game.get_encoded_state(state),device=self.model.device).unsqueeze(dim=0)
        )
        policy = torch.softmax(policy,dim=1).squeeze(dim=0).cpu().numpy()
        # Adding temperature for more exploration
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        # get the valid moves
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy)

        for search in range(self.args('num_searches')):
            node = root
            while node.is_fully_expended():
                node = node.select()

            value,is_terminated = self.game.get_value_and_terminated(action = node.action_taken,state = node.state)
            value = self.game.get_opponent_value(value)

            # Expand
            if not is_terminated :
                policy,val = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state),device=node.model.device).unsqueeze(dim=0)
                )
                policy = torch.softmax(policy,dim=1).squeeze(dim=0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                value = val.item()

                node.expand(policy)
            node.backpropagate(value)
        
        actions_probs = np.zeros(self.game.action_size)
        for child in root.children:
            actions_probs[child.action_taken] += child.visit_count
        actions_probs /= np.sum(actions_probs)
        return actions_probs
