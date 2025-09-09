import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from itertools import count
import math

from card import Card
from game_state import GameState, Player
from noble import Noble
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

GEM_ORDER = ["Diamond", "Sapphire", "Emerald", "Ruby", "Onyx"]
GEM_INDEX = {g: i for i, g in enumerate(GEM_ORDER)}
GEMS_WITH_GOLD = GEM_ORDER + ["gold"]          # bank & player tokens use gold too
TABLE_ROWS, TABLE_COLS = 3, 4              # 12 visible cards
TABLE_SLOTS = TABLE_ROWS * TABLE_COLS

# Sensible caps for normalisation
MAX_RESERVED = 3.0
MAX_TOKENS_PER_GEM = 7.0                      # 7 gems per gemtype in a 4 man game
MAX_GOLD = 5.0                                # most gold any 1 player can have
MAX_VP_PER_CARD = 5.0                         # best t3 cards give up to 5 VP
MAX_BONUS_PER_COLOR  = 18.0                   # from cards (18 onyx cards total from all 3 tierd decks)
MAX_CARD_COST        = 17.0                   # sum of gems for most expensive card
MAX_POINTS = 18.0                             # player has 14 points, buys 5p card.
MAX_COST_PER_COLOUR = 7.0                     # on 14, buys a 5 point card next turn.


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
    

def encode_card(card) -> np.ndarray:
    # 5 one-hot-gemType + 1 vp + 5 costs (len = 11)
    if card is None:
        return np.zeros(11, dtype=np.float32)

    v = np.zeros(11, dtype=np.float32)

    # One-hot gemType (bonus type)
    v[GEM_INDEX[card.gemType]] = 1.0

    # Victory points
    v[5] = card.victoryPoints / MAX_VP_PER_CARD   # scale 0–5 into 0–1

    # Costs (always in the same order: emerald, diamond, sapphire, onyx, ruby)
    costs = [card.cost[g] for g in GEM_ORDER]
    v[6:] = np.array(costs, dtype=np.float32) / MAX_COST_PER_COLOUR
    return v




def encode_player(p: Player) -> np.ndarray:
    # Output layout (len = 46):
    # [0]              points / MAX_POINTS
    # [1:6]            tokens (Diamond..Onyx) / MAX_TOKENS_PER_GEM
    # [6]              gold tokens / MAX_GOLD
    # [7:12]           bonuses (Diamond..Onyx) / MAX_BONUS_PER_COLOR
    # [12]             reserved_count / MAX_RESERVED
    # [13:46]          reserved cards (3 × encode_card), flattened
    pts = np.array([p.points / MAX_POINTS], dtype=np.float32)

    tok_cols = np.array([p.tokens.get(g, 0) for g in GEM_ORDER], dtype=np.float32) / np.float32(MAX_TOKENS_PER_GEM)
    tok_gold = np.array([p.tokens.get("gold", 0)], dtype=np.float32) / np.float32(MAX_GOLD)

    bon_cols = np.array([p.bonuses.get(g, 0) for g in GEM_ORDER], dtype=np.float32) / np.float32(MAX_BONUS_PER_COLOR)

    rc = min(len(p.reserved), int(MAX_RESERVED)) / np.float32(MAX_RESERVED)
    rc_arr = np.array([rc], dtype=np.float32)

    rvecs = []
    for i in range(int(MAX_RESERVED)):
        card = p.reserved[i] if i < len(p.reserved) else None
        rvecs.append(encode_card(card))  # each (11,)
    reserved_vec = np.concatenate(rvecs, axis=0).astype(np.float32)  # (33,)

    return np.concatenate([pts, tok_cols, tok_gold, bon_cols, rc_arr, reserved_vec], axis=0)

def encode_state():
    print("!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    

if __name__ == '__main__':
    main()