import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from itertools import count
import math
import argparse
import os

from card import Card
from game_state import GameState, Player
from noble import Noble
from typing import List, Dict
from loader import load_initial_state, load_valid_str_moves
from move_parser import parse_move
from engine import check_all_available_moves, apply_move
Transition = namedtuple('Transition', ('state', 'action', 'nextState', 'reward', 'done'))

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 20_000          # step-based decay (self-play)
EPS_DECAY_EPISODES = 500    # episode-based decay (hero mode): epsilon ~0.12 by ep 1000
TAU = 0.005
LR = 1e-4
Q_TARGET_CLIP = 10.0  # clip TD targets to [-Q_TARGET_CLIP, Q_TARGET_CLIP] to prevent Q-value explosion
REPLAY_CAPACITY = 50_000
NUM_EPISODES = 10_000
LOG_EVERY_EPISODES = 50
CHECKPOINT_EVERY_EPISODES = 100      # save hero at 100, 200, ... + bestwr when recent_win% beats previous best
CHECKPOINT_DIR = "checkpoints"
BESTWR_FILENAME = "hero_bestwr.pt"
OPPONENT_EPSILON = 0.1              # unused (no snapshot opponents)
CAP_LOSS_REWARD = -1.0               # terminal reward when game hits turn cap (hero "loses")
SKIP_PENALTY = -0.5                  # reward when hero has to skip (no other legal move); encourages avoiding dead states
MAX_TURNS_PER_GAME = 500             # lower cap so stalemates end sooner and cap-loss signal is stronger
EARLY_WIN_TURN_THRESHOLD = 350       # wins at or before this many turns get a bonus
EARLY_WIN_BONUS = 0.5                # extra reward for winning quickly
TERMINAL_WIN_REWARD = 2.0             # reward for winning (stronger signal than ±1)
TERMINAL_LOSS_REWARD = -2.0          # reward for losing
REWARD_CLIP = 3.0                     # clip reward to [-REWARD_CLIP, REWARD_CLIP] so ±2 terminal fits
SPARSE_REWARD = True                  # if True: only terminal win/loss ±2 + tiny time penalty; no VP/card-value shaping

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
MAX_BONUS_PER_COLOR  = 18.0                   # from cards (18 onyx cards total from all 3 decks)
MAX_CARD_COST        = 17.0                   # sum of gems for most expensive card
MAX_POINTS = 19.0                             # player has 14 points, buys 5p card.
MAX_COST_PER_COLOUR = 7.0                     # on 14, buys a 5 point card next turn.
MAX_REQ_PER_COLOR = 4.0                       # nobles require up to 4 of a colour
MAX_NOBLES = 5                                # max nobles on the table

N_PLAYERS = 2
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

    gems = [bank.get(g.lower(), 0) / gem_cap for g in GEM_ORDER]
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

    tok_cols = np.array([p.tokens.get(g.lower(), 0) for g in GEM_ORDER], dtype=np.float32) / np.float32(MAX_TOKENS_PER_GEM)
    tok_gold = np.array([p.tokens.get("gold", 0)], dtype=np.float32) / np.float32(MAX_GOLD)

    bon_cols = np.array([p.bonuses.get(g.lower(), 0) for g in GEM_ORDER], dtype=np.float32) / np.float32(MAX_BONUS_PER_COLOR)

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

def encode_state(state: GameState, turn: int = None) -> np.ndarray:
    num_players = len(state.players)

    parts = []
    parts.append(encode_table(state))                         # 132
    parts.append(encode_bank(state.bank, num_players))        # 6 (scaled by players)
    for p in state.players:                                   # n × 46 if n players (min 2)
        parts.append(encode_player(p))
    parts.append(encode_nobles(state.nobles, num_players))    # (num_players+1) * (5 + num_players)
    if turn is not None:
        parts.append(np.array([turn / MAX_TURNS_PER_GAME], dtype=np.float32))  # time remaining signal

    return np.concatenate(parts, axis=0).astype(np.float32)

def optimize_model(device, policy_net, target_net, optimizer, memory, criterion):
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.from_numpy(np.stack(batch.state)).float().to(device)
    action_batch = torch.cat(batch.action).to(device)  # [B, 1]
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)  # [B, 1]
    non_final_mask = torch.tensor([not d for d in batch.done], device=device, dtype=torch.bool)

    q_values = policy_net(state_batch).gather(1, action_batch)  # [B, 1]

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_mask.any():
        non_final_next_states = torch.from_numpy(
            np.stack([s for s, d in zip(batch.nextState, batch.done) if not d])
        ).float().to(device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_q_values = reward_batch + GAMMA * next_state_values.unsqueeze(1)
    # Clip targets to prevent Q-value explosion in long episodes (stable DQN)
    expected_q_values = expected_q_values.clamp(-Q_TARGET_CLIP, Q_TARGET_CLIP)

    loss = criterion(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    optimizer.step()

    # Soft target update
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

    return loss.item()


def main_self_play(num_episodes: int = NUM_EPISODES):
    """Self-play mode: one shared policy controls all seats and learns from all turns."""
    global ACTION_STRINGS, ACTIONS_PARSED, N_ACTIONS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ACTION_STRINGS = load_valid_str_moves()
    if not ACTION_STRINGS:
        raise FileNotFoundError("data/possible_moves.txt not found; cannot load action list")
    ACTIONS_PARSED = [canon_action(parse_move(s)) for s in ACTION_STRINGS]
    N_ACTIONS = len(ACTION_STRINGS)

    initialState = load_initial_state(N_PLAYERS)
    encoded_state = encode_state(initialState, turn=0)
    n_observations = len(encoded_state)

    policy_net = DQN(n_observations, N_ACTIONS).to(device)
    target_net = DQN(n_observations, N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_CAPACITY)
    criterion = nn.SmoothL1Loss()

    # Diagnostics
    running_loss_ema = None
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    player0_wins = 0
    games_ended = 0

    for episode in range(num_episodes):
        state = load_initial_state(N_PLAYERS)
        episode_reward = 0.0
        turn = 0

        while not state.game_over and turn < MAX_TURNS_PER_GAME:
            active_idx = state.active_idx
            prev_points = state.players[active_idx].points
            prev_tokens = dict(state.players[active_idx].tokens)
            prev_bonuses = dict(state.players[active_idx].bonuses)

            action, state_vec = selectAction(state, policy_net, device, turn=turn)
            action_idx = action.squeeze().item()

            step_env(state, action_idx)
            done = state.game_over
            next_state_vec = encode_state(state, turn=turn + 1)
            reward = compute_reward(prev_points, prev_tokens, prev_bonuses, state, active_idx, done)
            if action_idx < len(ACTIONS_PARSED) and ACTIONS_PARSED[action_idx][0] == "SKIP":
                reward += SKIP_PENALTY

            episode_reward += reward
            memory.push(
                state_vec,
                action.detach().cpu(),
                next_state_vec,
                reward,
                done,
            )
            loss_val = optimize_model(device, policy_net, target_net, optimizer, memory, criterion)
            if loss_val is not None:
                running_loss_ema = loss_val if running_loss_ema is None else 0.99 * running_loss_ema + 0.01 * loss_val
            turn += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(turn)
        if state.game_over and state.final_summary:
            games_ended += 1
            scores = [state.players[i].points for i in range(len(state.players))]
            max_score = max(scores)
            winners = [i for i in range(len(scores)) if scores[i] == max_score]
            if 0 in winners and len(winners) == 1:
                player0_wins += 1

        if (episode + 1) % LOG_EVERY_EPISODES == 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
            win_rate = (player0_wins / games_ended * 100) if games_ended else 0.0
            loss_str = f"loss_ema={running_loss_ema:.8f}" if running_loss_ema is not None else "loss_ema=N/A"
            print(
                f"[SELF] Ep {episode + 1:5d} | {loss_str} | avg_reward={avg_reward:.3f} | "
                f"avg_len={avg_length:.1f} | ended={games_ended} | P0_win%={win_rate:.1f}"
            )


def main_hero_vs_random(num_episodes: int = NUM_EPISODES, hero_idx: int = 0):
    """Hero-vs-opponents mode: only hero seat learns; others play random/opponent moves."""
    global ACTION_STRINGS, ACTIONS_PARSED, N_ACTIONS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ACTION_STRINGS = load_valid_str_moves()
    if not ACTION_STRINGS:
        raise FileNotFoundError("data/possible_moves.txt not found; cannot load action list")
    ACTIONS_PARSED = [canon_action(parse_move(s)) for s in ACTION_STRINGS]
    N_ACTIONS = len(ACTION_STRINGS)

    initialState = load_initial_state(N_PLAYERS)
    encoded_state = encode_state(initialState, turn=0)
    n_observations = len(encoded_state)

    policy_net = DQN(n_observations, N_ACTIONS).to(device)
    target_net = DQN(n_observations, N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_CAPACITY)
    criterion = nn.SmoothL1Loss()

    running_loss_ema = None
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    hero_wins = 0
    games_ended = 0
    RECENT_WIN_WINDOW = 100  # rolling window for "recent" win % (reflects current policy)
    recent_outcomes: deque = deque(maxlen=RECENT_WIN_WINDOW)  # 1=hero win, 0=loss, only when game ended

    # No snapshot opponents: hero vs random only (opponents list left empty)
    opponents: List[DQN] = []

    best_recent_win = -1.0  # best recent_win% so far; save hero_bestwr.pt when we beat it

    # Diagnostics: hero action-type distribution (rolling last N episodes)
    hero_action_history: deque = deque(maxlen=50)
    _diag_time_warned = False

    for episode in range(num_episodes):
        state = load_initial_state(N_PLAYERS)
        episode_reward = 0.0
        turn = 0
        last_hero_state_vec = None
        last_hero_action = None
        hero_actions_this_ep: Dict[str, int] = {"BUY": 0, "TAKE_3": 0, "TAKE_2": 0, "RESERVE": 0, "OTHER": 0, "buy_legal_turns": 0, "buy_chosen": 0}

        while not state.game_over and turn < MAX_TURNS_PER_GAME:
            active_idx = state.active_idx

            if active_idx == hero_idx:
                if not _diag_time_warned:
                    _diag_time_warned = True
                    print("[DIAG] State now includes turn/MAX_TURNS so policy can condition on time remaining.")
                    print("[DIAG] Cap loss is applied only to the last transition; gamma^steps is tiny so early moves get almost no gradient from 'hit cap'.")

                prev_points = state.players[active_idx].points
                prev_tokens = dict(state.players[active_idx].tokens)
                prev_bonuses = dict(state.players[active_idx].bonuses)

                # Was BUY legal this turn? (so we can report "buy rate when buy was legal")
                mask_np = legal_action_mask(state)
                buy_legal = any(ACTIONS_PARSED[i][0] == "BUY" for i in range(N_ACTIONS) if mask_np[i])
                if buy_legal:
                    hero_actions_this_ep["buy_legal_turns"] += 1

                # Episode-based epsilon so hero is mostly greedy by ~1k episodes (not step-based)
                eps_hero = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * episode / EPS_DECAY_EPISODES)
                action, state_vec = selectAction(state, policy_net, device, turn=turn, epsilon_override=eps_hero)
                action_idx = action.squeeze().item()
                action_cpu = action.detach().cpu()
                kind = ACTIONS_PARSED[action_idx][0] if action_idx < len(ACTIONS_PARSED) else "OTHER"
                hero_actions_this_ep[kind] = hero_actions_this_ep.get(kind, 0) + 1
                if kind == "BUY":
                    hero_actions_this_ep["buy_chosen"] += 1

                step_env(state, action_idx)
                done = state.game_over
                next_state_vec = encode_state(state, turn=turn + 1)
                reward = compute_reward(
                    prev_points, prev_tokens, prev_bonuses, state, active_idx, done, turn_count=turn + 1
                )
                if kind == "SKIP":
                    reward += SKIP_PENALTY

                episode_reward += reward
                memory.push(
                    state_vec,
                    action_cpu,
                    next_state_vec,
                    reward,
                    done,
                )
                last_hero_state_vec = state_vec
                last_hero_action = action_cpu
                loss_val = optimize_model(device, policy_net, target_net, optimizer, memory, criterion)
                if loss_val is not None:
                    running_loss_ema = loss_val if running_loss_ema is None else 0.99 * running_loss_ema + 0.01 * loss_val
            else:
                # Opponent: use a frozen snapshot policy if available, with its own epsilon-random moves
                if opponents:
                    opp_net = random.choice(opponents)
                    if random.random() < OPPONENT_EPSILON:
                        opp_action_idx = sample_random_legal_action_index(state)
                    else:
                        opp_action_idx = select_greedy_action_with_net(state, opp_net, device, turn=turn)
                else:
                    opp_action_idx = sample_random_legal_action_index(state)
                step_env(state, opp_action_idx)

            turn += 1

        # Cap = loss: if we hit max turns without game over, give hero a terminal loss signal
        if turn >= MAX_TURNS_PER_GAME and not state.game_over and last_hero_state_vec is not None:
            final_state_vec = encode_state(state, turn=turn)
            memory.push(
                last_hero_state_vec,
                last_hero_action,
                final_state_vec,
                CAP_LOSS_REWARD,
                True,
            )
            episode_reward += CAP_LOSS_REWARD
            loss_val = optimize_model(device, policy_net, target_net, optimizer, memory, criterion)
            if loss_val is not None:
                running_loss_ema = loss_val if running_loss_ema is None else 0.99 * running_loss_ema + 0.01 * loss_val

        hero_action_history.append(hero_actions_this_ep)
        episode_rewards.append(episode_reward)
        episode_lengths.append(turn)
        if state.game_over and state.final_summary:
            games_ended += 1
            scores = [state.players[i].points for i in range(len(state.players))]
            max_score = max(scores)
            winners = [i for i in range(len(scores)) if scores[i] == max_score]
            hero_won = hero_idx in winners and len(winners) == 1
            if hero_won:
                hero_wins += 1
            recent_outcomes.append(1 if hero_won else 0)

        # Save checkpoint every CHECKPOINT_EVERY_EPISODES (100, 200, ..., 1000)
        if (episode + 1) % CHECKPOINT_EVERY_EPISODES == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            path = os.path.join(CHECKPOINT_DIR, "hero_ep%05d.pt" % (episode + 1))
            torch.save({
                "policy": policy_net.state_dict(),
                "n_obs": n_observations,
                "n_actions": N_ACTIONS,
                "episode": episode + 1,
            }, path)
            print("[HERO] Saved checkpoint: %s" % path)

        # Best win-rate checkpoint: save when recent_win% beats previous best (at each 100-ep boundary)
        if (episode + 1) % CHECKPOINT_EVERY_EPISODES == 0 and recent_outcomes:
            recent_win_pct = 100.0 * sum(recent_outcomes) / len(recent_outcomes)
            if recent_win_pct > best_recent_win:
                best_recent_win = recent_win_pct
                best_path = os.path.join(CHECKPOINT_DIR, BESTWR_FILENAME)
                torch.save({
                    "policy": policy_net.state_dict(),
                    "n_obs": n_observations,
                    "n_actions": N_ACTIONS,
                    "episode": episode + 1,
                    "recent_win_pct": best_recent_win,
                }, best_path)
                print("[HERO] New best recent_win%%=%.1f -> %s" % (best_recent_win, best_path))

        if (episode + 1) % LOG_EVERY_EPISODES == 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
            win_rate_cumul = (hero_wins / games_ended * 100) if games_ended else 0.0
            recent_win = (100.0 * sum(recent_outcomes) / len(recent_outcomes)) if recent_outcomes else 0.0
            loss_str = f"loss_ema={running_loss_ema:.4f}" if running_loss_ema is not None else "loss_ema=N/A"
            # Buy rate when BUY was legal (not confounded by game length)
            total_buy_legal = sum(d.get("buy_legal_turns", 0) for d in hero_action_history)
            total_buy_chosen = sum(d.get("buy_chosen", 0) for d in hero_action_history)
            buy_rate_when_legal = (100.0 * total_buy_chosen / total_buy_legal) if total_buy_legal else 0.0
            avg_buys_per_ep = total_buy_chosen / len(hero_action_history) if hero_action_history else 0.0
            # recent_win% = performance metric; loss_ema = TD error (often not aligned with win rate)
            print(
                f"[HERO] Ep {episode + 1:5d} | recent_win%={recent_win:.1f} (last {len(recent_outcomes)}) | "
                f"cumul_win%={win_rate_cumul:.1f} | {loss_str} | avg_reward={avg_reward:.3f} | avg_len={avg_length:.1f}"
            )
            # print("       when BUY legal: chose BUY %.1f%% (avg %.1f buys/ep)" % (buy_rate_when_legal, avg_buys_per_ep))



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
    

def legal_action_mask(state: GameState) -> np.ndarray:
    """Boolean mask aligned with ACTIONS_PARSED (True = legal now)."""
    legal_now = {canon_action(a) for a in check_all_available_moves(state)}
    return np.array([a in legal_now for a in ACTIONS_PARSED], dtype=bool)


def sample_random_legal_action_index(state: GameState) -> int:
    """Sample a random legal action index (fallback to any index if none)."""
    mask_np = legal_action_mask(state)
    legal_idx = np.flatnonzero(mask_np)
    if legal_idx.size:
        return int(np.random.choice(legal_idx))
    return int(np.random.randint(0, N_ACTIONS))


def select_greedy_action_with_net(state: GameState, net: DQN, device: torch.device, turn: int = None) -> int:
    """Greedy action from a given network (used by frozen opponents)."""
    state_vec = encode_state(state, turn=turn)
    state_t = torch.from_numpy(state_vec).to(device).unsqueeze(0)
    mask_np = legal_action_mask(state)
    mask_t = torch.from_numpy(mask_np).to(device)
    with torch.no_grad():
        q = net(state_t)
        if mask_t.any():
            q[0, ~mask_t] = -float("inf")
            a = q.argmax(dim=1, keepdim=True)
        else:
            a = torch.randint(0, N_ACTIONS, (1, 1), device=device)
    return int(a.item())

def selectAction(state: GameState, policyNet: DQN, device: torch.device, turn: int = None, epsilon_override: float = None):
    """ε-greedy action selection returning both action index tensor and encoded state vector.

    If epsilon_override is set (e.g. hero mode with episode-based decay), use it and do not
    increment the global steps counter. Otherwise use step-based decay.
    Returns:
        action_tensor: shape [1,1], long
        state_vec: numpy float32 vector encoding the state (for replay)
    """
    global steps

    state_vec = encode_state(state, turn=turn)  # numpy float32
    state_t = torch.from_numpy(state_vec).to(device).unsqueeze(0)

    if epsilon_override is not None:
        eps_threshold = epsilon_override
    else:
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
        return a.to(torch.long), state_vec
    else:
        # If we rolled below the epsilon threshold, do something random for exploration
        idx = sample_random_legal_action_index(state)
        return torch.tensor([[idx]], device=device, dtype=torch.long), state_vec
    


def step_env(state: GameState, action_index: int):
    """Map index -> engine move tuple, apply, return (next_state, info_str, done)."""
    move = ACTIONS_PARSED[action_index]   # canonical engine tuple
    info = apply_move(state, move)        # mutates state; returns status string
    done = state.game_over
    return state, info, done


def _card_progress_value(card: Card, tokens: Dict[str, int], bonuses: Dict[str, int]) -> float:
    """Heuristic value: how close the player is to affording this card."""
    if card is None:
        return 0.0
    # Compute remaining total cost after bonuses and coloured tokens (ignore gold for simplicity)
    remaining = 0
    for gem in GEM_ORDER:
        cost = card.cost[gem]
        bonus = bonuses.get(gem.lower(), 0)
        need = max(0, cost - bonus)
        have = tokens.get(gem.lower(), 0)
        remaining += max(0, need - have)
    # Higher VP and smaller remaining cost => higher value
    return float(card.victoryPoints) / (1.0 + float(remaining))


def _state_card_value_for_player(state: GameState, player_idx: int) -> float:
    """Sum progress value over table + reserved cards for a given player."""
    p = state.players[player_idx]
    tokens = p.tokens
    bonuses = p.bonuses
    total = 0.0
    # Visible table cards
    for c in list(state.table_t1) + list(state.table_t2) + list(state.table_t3):
        total += _card_progress_value(c, tokens, bonuses)
    # Reserved cards
    for c in p.reserved:
        total += _card_progress_value(c, tokens, bonuses)
    return total


def compute_reward(
    prev_points: int,
    prev_tokens: Dict[str, int],
    prev_bonuses: Dict[str, int],
    state_after: GameState,
    active_player_idx: int,
    done: bool,
    turn_count: int = None,
) -> float:
    """Reward: terminal win/loss ±2 (+ optional early-win bonus). If not SPARSE_REWARD, add VP/card-value/bonus shaping."""
    reward = 0.0

    if not SPARSE_REWARD:
        p = state_after.players[active_player_idx]
        delta_points = p.points - prev_points
        reward += delta_points * 0.5
        before_value = _state_card_value_for_player(state_after, active_player_idx)
        current_tokens, current_bonuses = p.tokens, p.bonuses
        p.tokens = prev_tokens
        p.bonuses = prev_bonuses
        try:
            prev_value = _state_card_value_for_player(state_after, active_player_idx)
        finally:
            p.tokens = current_tokens
            p.bonuses = current_bonuses
        reward += 0.1 * (before_value - prev_value)
        bonus_delta = sum(p.bonuses.get(g.lower(), 0) - prev_bonuses.get(g.lower(), 0) for g in GEM_ORDER)
        reward += 0.02 * bonus_delta

    # Terminal: win/loss (strong ±2 signal)
    if done and state_after.final_summary:
        p = state_after.players[active_player_idx]
        scores = [state_after.players[i].points for i in range(len(state_after.players))]
        max_score = max(scores)
        winners = [i for i in range(len(scores)) if scores[i] == max_score]
        if active_player_idx in winners:
            reward += TERMINAL_WIN_REWARD if len(winners) == 1 else 0.0
            if not SPARSE_REWARD and turn_count is not None and len(winners) == 1 and turn_count <= EARLY_WIN_TURN_THRESHOLD:
                reward += EARLY_WIN_BONUS
        else:
            reward += TERMINAL_LOSS_REWARD

    reward -= 0.005  # small time penalty
    return float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))


def main():
    parser = argparse.ArgumentParser(description="Train Splendor DQN")
    parser.add_argument(
        "--mode",
        choices=["self_play", "hero_vs_random"],
        default="self_play",
        help="Training mode: shared self-play or hero vs random opponents",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help="Number of training episodes (games) to run",
    )
    args = parser.parse_args()

    if args.mode == "self_play":
        main_self_play(num_episodes=args.episodes)
    else:
        main_hero_vs_random(num_episodes=args.episodes)


if __name__ == '__main__':
    main()