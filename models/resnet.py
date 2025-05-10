import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = num_hidden,out_channels = num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(in_channels = num_hidden,out_channels = num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self,x):
        residual = x
        x = nn.ReLU(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = nn.ReLU(x)
        return x
    
class ResNet(nn.Module):

    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device

        self.startBlock = nn.Sequential(
                                    nn.conv2d(in_channels=3,
                                              out_channels=num_hidden,
                                              kernel_size = 3,
                                              padding = 1),
                                    nn.BatchNorm2d(num_hidden),
                                    nn.ReLU()
                                    )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        

        # Policy Network
        self.policyHead = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32*game.n_rows*game.n_cols, out_features=game.action_size)
        )

        # Value Network
        self.valueHead = nn.Sequential(
            nn.Conv2d(in_channels=num_hidden,out_channels=3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=3*game.n_rows*game.n_cols,out_features=1),
            nn.Tanh()
        )

        self.to(device)



    def forward(self,x):
        x = self.startBlock(x)
        for resblock in self.backBone:
            x = resblock(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
