import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from itertools import count
import math

from card import Card
from typing import List, Dict

Transition = namedtuple('Transition', ('state', 'action', 'nextState', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

N_POSSIBLE_ACTIONS = 42

GEMS = ["diamond", "sapphire", "emerald", "ruby", "onyx"]
GEMS_WITH_GOLD = GEMS + ["gold"]          # bank & player tokens use gold too
MAX_RESERVED = 3
TABLE_ROWS, TABLE_COLS = 3, 4              # 12 visible cards
TABLE_SLOTS = TABLE_ROWS * TABLE_COLS

# Sensible caps for normalisation
MAX_TOKENS_PER_GEM = 10                  
MAX_GOLD = 5
MAX_BONUS_PER_COLOR  = 18                   # from cards
MAX_CARD_COST        = 17
MAX_POINTS           = 19                   # on 14, buys a 5 point card next turn.


# following some tutorial.

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# cyclic buffer of bounded size that holds state transitions observed recently.
# sample method is random to ensure the batch is decorrelated
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batchSize):
        """Get a random sample of observed transitions."""
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)
    

def encode_card(card: Card):
    print("!")

def encode_state():
    print("!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    

if __name__ == '__main__':
    main()