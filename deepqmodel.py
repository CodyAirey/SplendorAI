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
from loader import load_initial_state, load_valid_str_moves
from move_parser import parse_move
from engine import check_all_available_moves, apply_move
Transition = namedtuple('Transition', ('state', 'action', 'nextState', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

GEM_ORDER = ["Diamond", "Sapphire", "Emerald", "Ruby", "Onyx"]
GEM_ORDER_LETTERS = ["D", "S", "E", "R", "O"]
GEM_ORDER_LETTER_INDEX = {g: i for i, g in enumerate(GEM_ORDER_LETTERS)}
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
MAX_REQ_PER_COLOR = 4.0                       # nobles require up to 4 of a colour
MAX_NOBLES = 5                                # max nobles on the table

N_PLAYERS = 4
steps = 0  # global env step counter for epsilon schedule


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

def encode_table(state: GameState) -> np.ndarray:
    # Flatten 3×4 table into a single vector (len = 12*11 = 132)
    vecs = []
    for row in (state.table_t1, state.table_t2, state.table_t3):
        cards = list(row)[:TABLE_COLS]
        while len(cards) < TABLE_COLS:   # pad if short
            cards.append(None)
        for c in cards:
            vecs.append(encode_card(c))
    return np.concatenate(vecs, axis=0).astype(np.float32)

def encode_bank(bank: dict, num_players: int) -> np.ndarray:
    # Order: Diamond, Sapphire, Emerald, Ruby, Onyx, Gold
    # Per-colour cap depends on player count (gold is always /5)
    if num_players == 2:
        gem_cap = 4.0
    elif num_players == 3:
        gem_cap = 5.0
    elif num_players == 4:
        gem_cap = 7.0
    else:
        raise ValueError("num_players must be 2, 3, or 4")

    gems = [bank.get(g, 0) / gem_cap for g in GEM_ORDER]
    gold = bank.get("gold", 0) / MAX_GOLD
    return np.array(gems + [gold], dtype=np.float32)

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


def encode_noble(noble: Noble, n_players: int) -> np.ndarray:
    # length = 5 (requirements) + n_players (owner one-hot)
    if noble is None:
        return np.zeros(5 + n_players, dtype=np.float32)

    # requirements
    reqs = np.array([
        noble.diamond,
        noble.sapphire,
        noble.emerald,
        noble.ruby,
        noble.onyx,
    ], dtype=np.float32) / MAX_REQ_PER_COLOR

    # ownership one-hot
    owner = np.zeros(n_players, dtype=np.float32)
    if 0 <= noble.playerVisited < n_players:
        owner[noble.playerVisited] = 1.0

    return np.concatenate([reqs, owner], axis=0)

def encode_nobles(nobles: list, num_players: int) -> np.ndarray:
    # Always num_players + 1 nobles visible (pad with None)
    max_nobles = num_players + 1
    vecs = []
    for i in range(max_nobles):
        n = nobles[i] if i < len(nobles) else None
        vecs.append(encode_noble(n, num_players))
    return np.concatenate(vecs, axis=0).astype(np.float32)

def encode_state(state: GameState) -> np.ndarray:
    num_players = len(state.players)

    parts = []
    parts.append(encode_table(state))                         # 132
    parts.append(encode_bank(state.bank, num_players))        # 6 (scaled by players)
    for p in state.players:                                   # n × 46 if n players (min 2)
        parts.append(encode_player(p))
    parts.append(encode_nobles(state.nobles, num_players))    # (num_players+1) * (5 + num_players)
    # parts.append(encode_num_players(num_players)) # maybe? unsure.... 

    return np.concatenate(parts, axis=0).astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initialState = load_initial_state(N_PLAYERS)
    encoded_state = encode_state(initialState)

    ACTION_STRINGS = load_valid_str_moves()                 # e.g. ["B(0,0)", "R(0,0)", "T(DSR)", ...]
    ACTIONS_PARSED = [canon_action(parse_move(s)) for s in ACTION_STRINGS]  # tuples engine understands
    N_ACTIONS = len(ACTION_STRINGS)

    print(check_all_available_moves(initialState))

    n_observations = len(encoded_state)
    actions = load_valid_str_moves()

    policy_net = DQN(n_observations, N_ACTIONS).to(device)
    target_net = DQN(n_observations, N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)



def canon_action(a):
    #Canonicalise (kind, payload) to match engine ordering
    kind, payload = a
    kind = kind.upper()

    if kind == "TAKE_3":
        # Upper-case then sort by engine's letter order D,S,E,R,O
        letters = [g.upper() for g in payload]
        letters.sort(key=GEM_ORDER_LETTER_INDEX.get)
        return (kind, tuple(letters))

    if kind == "TAKE_2":
        return (kind, payload.upper())

    if kind in ("BUY", "RESERVE"):
        r, c = payload
        if isinstance(r, str):        # ("R", idx) form
            return (kind, (r.upper(), int(c)))
        return (kind, (int(r), int(c)))

    return (kind, payload)
    

def legal_action_mask(state: GameState, ACTIONS_PARSED: List[tuple]) -> np.ndarray:
    """Boolean mask aligned with ACTIONS_PARSED (True = legal now)."""
    legal_now = {canon_action(a) for a in check_all_available_moves(state)}
    return np.array([a in legal_now for a in ACTIONS_PARSED], dtype=bool)


def selectAction(state: GameState, policyNet: DQN, device: torch.device, N_ACTIONS: int):
    global steps

    state_vec = encode_state(state)
    state_t = torch.from_numpy(state_vec).to(device).unsqueeze(0)

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
    steps += 1

    mask_np = legal_action_mask(state)                       # [N_ACTIONS] bool
    mask_t  = torch.from_numpy(mask_np).to(device)
    
    # If we rolled a number above the epsilon threshold, do what is determined by our learned model
    if random.random() > eps_threshold:
        with torch.no_grad(): #Turn off gradient flowing, the learning happens from the transition stores
            q = policyNet(state_t)                          # [1, N_ACTIONS]
            if mask_t.any():
                q[0, ~mask_t] = -float("inf")               # forbid illegal
                a = q.argmax(dim=1, keepdim=True)           # [1,1]
            else:
                a = torch.randint(0, N_ACTIONS, (1,1), device=device)
        return a.to(torch.long)
    else:
        # If we rolled below the epsilon threshold, do something random for exploration
        legal_idx = np.flatnonzero(mask_np)
        if legal_idx.size:
            idx = int(np.random.choice(legal_idx))
        else:
            idx = int(np.random.randint(0, N_ACTIONS))
        return torch.tensor([[idx]], device=device, dtype=torch.long)

if __name__ == '__main__':
    main()