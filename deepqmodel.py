"""
Splendor DQN: train a policy to play Splendor (hero vs random opponents).

Usage:
    python deepqmodel.py [--episodes N]
"""

import argparse
import math
import os
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from engine import apply_move, check_all_available_moves
from game_state import GameState, Player
from loader import load_initial_state, load_valid_str_moves
from move_parser import parse_move
from noble import Noble

# -----------------------------------------------------------------------------
# Replay transition (state, action, next_state, reward, done)
# -----------------------------------------------------------------------------
Transition = namedtuple("Transition", ("state", "action", "nextState", "reward", "done"))


# -----------------------------------------------------------------------------
# Training config: all tunable hyperparameters in one place
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """All training hyperparameters. Tweak these instead of scattered globals."""

    # DQN / optimizer
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    tau: float = 0.005  # soft target update
    q_target_clip: float = 10.0  # clip TD targets to avoid Q explosion

    # Exploration (epsilon-greedy, episode-based for hero)
    eps_start: float = 0.9       # when training from scratch (vs random)
    eps_start_resume: float = 0.6   # when vs checkpoint: start more random, then hone in (decay to eps_end)
    eps_end: float = 0.01
    eps_decay_episodes: float = 2000.0

    # Replay
    replay_capacity: int = 25_000
    terminal_replay_multiplier: int = 8  # oversample win/loss transitions

    # Monte Carlo: terminal outcome only (win/loss), target = full-episode return G_t
    mc_win_reward: float = 1.0
    mc_loss_reward: float = -1.0

    # Game
    max_turns_per_game: int = 500
    num_episodes: int = 60_000
    num_players: int = 2

    # Schedule: warmup (no training), then train every N episodes
    warmup_episodes: int = 3000
    train_every_n_episodes: int = 1000
    train_steps_per_phase: int = 5000
    # Vs checkpoint only: no warmup, fewer steps, lower LR, oversample wins (avoid overfitting / loss-dominated buffer)
    train_steps_per_phase_vs_checkpoint: int = 2000
    lr_vs_checkpoint: float = 2.5e-5  # lower than lr so we don't overwrite the good policy
    win_oversample_vs_checkpoint: int = 3  # push each transition from winning games this many times

    # Logging / checkpoints (save names use hero_vs_random_* or hero_vs_checkpoint_* from CLI)
    log_every_episodes: int = 1000
    checkpoint_every_episodes: int = 1000
    checkpoint_dir: str = "checkpoints"


# -----------------------------------------------------------------------------
# Game / encoding constants (Splendor rules, not training knobs)
# -----------------------------------------------------------------------------
GEM_ORDER = ["Diamond", "Sapphire", "Emerald", "Ruby", "Onyx"]
GEM_ORDER_LETTERS = ["D", "S", "E", "R", "O"]
GEM_ORDER_LETTER_INDEX = {g: i for i, g in enumerate(GEM_ORDER_LETTERS)}
GEM_INDEX = {g: i for i, g in enumerate(GEM_ORDER)}
TABLE_ROWS, TABLE_COLS = 3, 4

MAX_RESERVED = 3.0
MAX_TOKENS_PER_GEM = 7.0
MAX_GOLD = 5.0
MAX_VP_PER_CARD = 5.0
MAX_BONUS_PER_COLOR = 18.0
MAX_POINTS = 19.0
MAX_COST_PER_COLOUR = 7.0
MAX_REQ_PER_COLOR = 4.0


# -----------------------------------------------------------------------------
# Action space: loaded once from possible_moves.txt, passed through the code
# -----------------------------------------------------------------------------
class ActionSpace:
    """Parsed action list and size. Created once at startup, no globals."""

    def __init__(self, action_strings: List[str], canon_fn):
        self.strings = action_strings
        self.parsed = [canon_fn(parse_move(s)) for s in action_strings]
        self.n_actions = len(action_strings)


# -----------------------------------------------------------------------------
# Model and replay buffer
# -----------------------------------------------------------------------------

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
    

# -----------------------------------------------------------------------------
# State encoding: game state -> fixed-size float vector for the DQN
# -----------------------------------------------------------------------------
def encode_card(card) -> np.ndarray:
    """5 one-hot gemType + 1 VP + 5 costs (len = 11)."""
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

def encode_state(state: GameState, turn: int = None, max_turns: int = 500) -> np.ndarray:
    num_players = len(state.players)
    parts = []
    parts.append(encode_table(state))
    parts.append(encode_bank(state.bank, num_players))
    for p in state.players:
        parts.append(encode_player(p))
    parts.append(encode_nobles(state.nobles, num_players))
    if turn is not None:
        parts.append(np.array([turn / max_turns], dtype=np.float32))
    return np.concatenate(parts, axis=0).astype(np.float32)

def optimize_model(
    device: torch.device,
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    memory: ReplayMemory,
    criterion: nn.Module,
    cfg: TrainConfig,
):
    """One DQN gradient step. Returns loss or None if buffer too small."""
    if len(memory) < cfg.batch_size:
        return None

    transitions = memory.sample(cfg.batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.from_numpy(np.stack(batch.state)).float().to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = policy_net(state_batch).gather(1, action_batch)
    # Monte Carlo: target is the stored return G_t (no bootstrap)
    expected_q = reward_batch.clamp(-cfg.q_target_clip, cfg.q_target_clip)

    loss = criterion(q_values, expected_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    optimizer.step()

    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(cfg.tau * policy_param.data + (1.0 - cfg.tau) * target_param.data)

    return loss.item()


# -----------------------------------------------------------------------------
# Main: hero vs random (only mode)
# -----------------------------------------------------------------------------
def _load_checkpoint(path: str, device: torch.device, n_observations: int, n_actions: int):
    """Load policy state dict; validate n_obs/n_actions. Returns state_dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path!r}")
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("n_obs") != n_observations or ckpt.get("n_actions") != n_actions:
        raise ValueError(
            f"Checkpoint {path} has n_obs={ckpt.get('n_obs')}, n_actions={ckpt.get('n_actions')}; "
            f"expected {n_observations}, {n_actions}."
        )
    return ckpt["policy"]

def run_hero_vs_random(
    cfg: TrainConfig,
    num_episodes: int = None,
    hero_idx: int = 0,
    start_path: str = None,
    opponent_path: str = None,
):
    """
    Train a single policy (hero). start_path = initial hero weights; opponent_path = fixed opponent (None = random).
    Saves as hero_vs_random_epXXXX / hero_vs_checkpoint_epXXXX and _bestwr.
    """
    num_episodes = num_episodes if num_episodes is not None else cfg.num_episodes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_turns = cfg.max_turns_per_game
    run_tag = "hero_vs_random" if opponent_path is None else "hero_vs_checkpoint"

    # Load action space once
    action_strings = load_valid_str_moves()
    if not action_strings:
        raise FileNotFoundError("data/possible_moves.txt not found")
    action_space = ActionSpace(action_strings, canon_action)

    initial_state = load_initial_state(cfg.num_players)
    n_observations = len(encode_state(initial_state, turn=0, max_turns=max_turns))

    policy_net = DQN(n_observations, action_space.n_actions).to(device)
    target_net = DQN(n_observations, action_space.n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    if start_path:
        policy_net.load_state_dict(_load_checkpoint(start_path, device, n_observations, action_space.n_actions))
        target_net.load_state_dict(policy_net.state_dict())
        print(f"[HERO] Start: {start_path}")
    else:
        print("[HERO] Start: random init")

    opponent_net = DQN(n_observations, action_space.n_actions).to(device)
    if opponent_path:
        opponent_net.load_state_dict(_load_checkpoint(opponent_path, device, n_observations, action_space.n_actions))
        print(f"[HERO] Opponent: fixed checkpoint {opponent_path}")
    else:
        opponent_net.load_state_dict(policy_net.state_dict())  # unused
        print("[HERO] Opponent: random")

    # Vs checkpoint: no warmup, fewer gradient steps, lower LR, oversample wins
    warmup_eff = 0 if opponent_path else cfg.warmup_episodes
    train_steps_eff = cfg.train_steps_per_phase_vs_checkpoint if opponent_path else cfg.train_steps_per_phase
    lr_eff = cfg.lr_vs_checkpoint if opponent_path else cfg.lr
    if opponent_path:
        print(f"[HERO] Vs checkpoint: warmup=0, train_steps={train_steps_eff}, lr={lr_eff}, win_oversample={cfg.win_oversample_vs_checkpoint}")

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr_eff, amsgrad=True)
    memory = ReplayMemory(cfg.replay_capacity)
    criterion = nn.SmoothL1Loss()

    # Tracking
    running_loss_ema = None
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    hero_wins = 0
    games_ended = 0
    recent_outcomes: deque = deque(maxlen=1000)  # recent_win% over last 1k games
    best_recent_win = -1.0
    hero_action_history: deque = deque(maxlen=50)

    for episode in range(num_episodes):
        state = load_initial_state(cfg.num_players)
        episode_reward = 0.0
        turn = 0
        last_hero_state_vec = None
        last_hero_action = None
        hero_trajectory: List[Tuple] = []  # (state_vec, action_cpu, next_state_vec, done) for MC
        hero_actions_this_ep: Dict[str, int] = {
            "BUY": 0, "TAKE_3": 0, "TAKE_2": 0, "RESERVE": 0, "OTHER": 0,
            "buy_legal_turns": 0, "buy_chosen": 0,
        }

        while not state.game_over and turn < max_turns:
            active_idx = state.active_idx

            if active_idx == hero_idx:
                mask_np = legal_action_mask(state, action_space.parsed)
                buy_legal = any(
                    action_space.parsed[i][0] == "BUY"
                    for i in range(action_space.n_actions)
                    if mask_np[i]
                )
                if buy_legal:
                    hero_actions_this_ep["buy_legal_turns"] += 1

                eps_start_eff = cfg.eps_start_resume if start_path else cfg.eps_start
                eps = cfg.eps_end + (eps_start_eff - cfg.eps_end) * math.exp(
                    -1.0 * episode / cfg.eps_decay_episodes
                )
                action, state_vec = select_action(
                    state, policy_net, device, action_space,
                    turn=turn, max_turns=max_turns, epsilon=eps,
                )
                action_idx = action.squeeze().item()
                action_cpu = action.detach().cpu()
                kind = action_space.parsed[action_idx][0] if action_idx < len(action_space.parsed) else "OTHER"
                hero_actions_this_ep[kind] = hero_actions_this_ep.get(kind, 0) + 1
                if kind == "BUY":
                    hero_actions_this_ep["buy_chosen"] += 1

                step_env(state, action_idx, action_space.parsed)
                done = state.game_over
                next_state_vec = encode_state(state, turn=turn + 1, max_turns=max_turns)
                hero_trajectory.append((state_vec, action_cpu, next_state_vec, done))
                last_hero_state_vec = state_vec
                last_hero_action = action_cpu
            else:
                if opponent_path is not None:
                    opp_idx = select_greedy_action_with_net(
                        state, opponent_net, device, action_space, turn=turn, max_turns=max_turns
                    )
                else:
                    opp_idx = sample_random_legal_action_index(state, action_space)
                step_env(state, opp_idx, action_space.parsed)

            turn += 1

        # Monte Carlo: outcome R, then back up return G_t along the trajectory
        if hero_trajectory:
            if turn >= max_turns and not state.game_over and last_hero_state_vec is not None:
                final_state_vec = encode_state(state, turn=turn, max_turns=max_turns)
                hero_trajectory.append((last_hero_state_vec, last_hero_action, final_state_vec, True))
                R = cfg.mc_loss_reward
            elif state.game_over and state.final_summary:
                scores = [state.players[i].points for i in range(len(state.players))]
                max_score = max(scores)
                winners = [i for i in range(len(scores)) if scores[i] == max_score]
                R = cfg.mc_win_reward if (hero_idx in winners and len(winners) == 1) else cfg.mc_loss_reward
            else:
                R = cfg.mc_loss_reward
            T = len(hero_trajectory)
            # Vs checkpoint: oversample winning trajectories so buffer isn't dominated by losses
            win_mult = cfg.win_oversample_vs_checkpoint if (opponent_path and R == cfg.mc_win_reward) else 1
            for t, (s, a, next_s, done) in enumerate(hero_trajectory):
                G_t = (cfg.gamma ** (T - 1 - t)) * R
                G_t = max(-cfg.q_target_clip, min(cfg.q_target_clip, G_t))
                base_mult = cfg.terminal_replay_multiplier if (t == T - 1) else 1
                for _ in range(base_mult * win_mult):
                    memory.push(s, a, next_s, float(G_t), done)
            episode_reward = R
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

        # Training phase: every N episodes after warmup
        if (episode + 1) >= warmup_eff and (episode + 1) % cfg.train_every_n_episodes == 0:
            for _ in range(train_steps_eff):
                loss_val = optimize_model(
                    device, policy_net, target_net, optimizer, memory, criterion, cfg,
                )
                if loss_val is not None:
                    running_loss_ema = (
                        loss_val if running_loss_ema is None
                        else 0.99 * running_loss_ema + 0.01 * loss_val
                    )
            if running_loss_ema is not None:
                print(f"[HERO] Training phase at ep {episode + 1}: {train_steps_eff} steps, loss_ema={running_loss_ema:.4f}")

        # Checkpoints: hero_vs_random_epXXXX or hero_vs_checkpoint_epXXXX (and _bestwr)
        if (episode + 1) % cfg.checkpoint_every_episodes == 0:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            path = os.path.join(cfg.checkpoint_dir, f"{run_tag}_ep{episode + 1:05d}.pt")
            torch.save({
                "policy": policy_net.state_dict(),
                "n_obs": n_observations,
                "n_actions": action_space.n_actions,
                "episode": episode + 1,
            }, path)
            print("[HERO] Saved checkpoint: %s" % path)

        if (episode + 1) % cfg.checkpoint_every_episodes == 0 and recent_outcomes:
            recent_win_pct = 100.0 * sum(recent_outcomes) / len(recent_outcomes)
            if recent_win_pct > best_recent_win:
                best_recent_win = recent_win_pct
                best_path = os.path.join(cfg.checkpoint_dir, f"{run_tag}_bestwr.pt")
                torch.save({
                    "policy": policy_net.state_dict(),
                    "n_obs": n_observations,
                    "n_actions": action_space.n_actions,
                    "episode": episode + 1,
                    "recent_win_pct": best_recent_win,
                }, best_path)
                print("[HERO] New best recent_win%%=%.1f -> %s" % (best_recent_win, best_path))

        # Log
        if (episode + 1) % cfg.log_every_episodes == 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
            avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
            win_rate_cumul = (hero_wins / games_ended * 100) if games_ended else 0.0
            recent_win = (100.0 * sum(recent_outcomes) / len(recent_outcomes)) if recent_outcomes else 0.0
            loss_str = "warmup (no train)" if episode < warmup_eff else f"loss_ema={running_loss_ema:.4f}"
            if warmup_eff and (episode + 1) == warmup_eff:
                print(f"[HERO] Warmup complete ({warmup_eff} episodes). Training started.")
            print(
                f"[HERO] Ep {episode + 1:5d} | recent_win%={recent_win:.1f} (last {len(recent_outcomes)}) | "
                f"cumul_win%={win_rate_cumul:.1f} | {loss_str} | avg_reward={avg_reward:.3f} | avg_len={avg_length:.1f}"
            )



# -----------------------------------------------------------------------------
# Action helpers: canonical move format, legal mask, selection, env step
# -----------------------------------------------------------------------------
def canon_action(a):
    """Canonicalise (kind, payload) to match engine ordering."""
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
    

def legal_action_mask(state: GameState, actions_parsed: List) -> np.ndarray:
    """Boolean mask aligned with actions_parsed (True = legal in this state)."""
    legal_now = {canon_action(a) for a in check_all_available_moves(state)}
    return np.array([a in legal_now for a in actions_parsed], dtype=bool)


def sample_random_legal_action_index(state: GameState, action_space: ActionSpace) -> int:
    """Random legal action index; fallback to any index if none legal."""
    mask_np = legal_action_mask(state, action_space.parsed)
    legal_idx = np.flatnonzero(mask_np)
    if legal_idx.size:
        return int(np.random.choice(legal_idx))
    return int(np.random.randint(0, action_space.n_actions))


def select_greedy_action_with_net(
    state: GameState,
    net: DQN,
    device: torch.device,
    action_space: ActionSpace,
    turn: int = None,
    max_turns: int = 500,
) -> int:
    """Greedy action from a given network (e.g. frozen opponent)."""
    state_vec = encode_state(state, turn=turn, max_turns=max_turns)
    state_t = torch.from_numpy(state_vec).to(device).unsqueeze(0)
    mask_np = legal_action_mask(state, action_space.parsed)
    mask_t = torch.from_numpy(mask_np).to(device)
    with torch.no_grad():
        q = net(state_t)
        if mask_t.any():
            q[0, ~mask_t] = -float("inf")
            a = q.argmax(dim=1, keepdim=True)
        else:
            a = torch.randint(0, action_space.n_actions, (1, 1), device=device)
    return int(a.item())


def select_action(
    state: GameState,
    policy_net: DQN,
    device: torch.device,
    action_space: ActionSpace,
    turn: int = None,
    max_turns: int = 500,
    epsilon: float = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Epsilon-greedy action. Returns (action_tensor [1,1], state_vec)."""
    state_vec = encode_state(state, turn=turn, max_turns=max_turns)
    state_t = torch.from_numpy(state_vec).to(device).unsqueeze(0)
    eps = epsilon if epsilon is not None else 0.01
    mask_np = legal_action_mask(state, action_space.parsed)
    mask_t = torch.from_numpy(mask_np).to(device)

    if random.random() > eps:
        with torch.no_grad():
            q = policy_net(state_t)
            if mask_t.any():
                q[0, ~mask_t] = -float("inf")
                a = q.argmax(dim=1, keepdim=True)
            else:
                a = torch.randint(0, action_space.n_actions, (1, 1), device=device)
        return a.to(torch.long), state_vec
    idx = sample_random_legal_action_index(state, action_space)
    return torch.tensor([[idx]], device=device, dtype=torch.long), state_vec


def step_env(state: GameState, action_index: int, actions_parsed: List) -> Tuple[GameState, str, bool]:
    """Apply move at index; mutates state. Returns (state, info_str, done)."""
    move = actions_parsed[action_index]
    info = apply_move(state, move)
    return state, info, state.game_over


def main():
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train Splendor DQN")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        help='Opponent: "random" or path to a .pt checkpoint (fixed opponent; saves as hero_vs_checkpoint_*)',
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Path to .pt to load as hero start. If unset and --opponent is a path, hero starts from that same file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=cfg.num_episodes,
        help="Number of games to run (default: %(default)s)",
    )
    args = parser.parse_args()

    opponent_path = None if (args.opponent.lower() == "random") else args.opponent
    start_path = args.start if args.start else opponent_path  # default start = opponent when vs checkpoint

    run_hero_vs_random(
        cfg,
        num_episodes=args.episodes,
        start_path=start_path,
        opponent_path=opponent_path,
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception:
        import traceback
        traceback.print_exc()
        raise