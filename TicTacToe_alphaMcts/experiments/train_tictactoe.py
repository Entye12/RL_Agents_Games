import torch
import torch.optim as optim

from games.tictactoe import TicTacToe
from models.resnet import ResNet
from agents.alphaZero import AlphaZero

import os

def main():
    # Training hyperparameters
    args = {
        'num_iterations': 10,
        'num_selfPlay_iterations': 25,
        'num_epochs': 5,
        'batch_size': 32,
        'C': 1.0,
        'num_searches': 25,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'temperature': 1.0,
        'num_resBlocks': 3,
        'num_hidden': 64,
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create save directory if needed
    os.makedirs('trained_models', exist_ok=True)

    # Game instance
    game = TicTacToe(n_rows=3, n_cols=3)

    # Model and optimizer
    model = ResNet(game=game,
                   num_resBlocks=args['num_resBlocks'],
                   num_hidden=args['num_hidden'],
                   device=args['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])

    # AlphaZero agent
    agent = AlphaZero(model=model, optimizer=optimizer, game=game, args=args)

    # Launch training
    print(f"Training on {args['device']}...")
    agent.learn()

if __name__ == '__main__':
    main()
