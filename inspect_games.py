"""
Inspect games played by a saved hero checkpoint.
Loads a checkpoint, runs N games with hero (greedy) vs random opponents,
and writes a trace file per game: turns, who moved, state summaries, actions, results.
Also saves a .replay file per game (initial state + move list + state_snapshots) for
UI replay: python initalize_ui.py traces/game_001.replay
"""
import argparse
import copy
import os
import pickle
import torch

from loader import load_initial_state, load_valid_str_moves
from move_parser import parse_move
from engine import apply_move, row_to_table_and_deck
from game_state import GameState
import deepqmodel as dq

dq.ACTION_STRINGS = load_valid_str_moves()
if not dq.ACTION_STRINGS:
    raise FileNotFoundError("data/possible_moves.txt not found")
dq.ACTIONS_PARSED = [dq.canon_action(parse_move(s)) for s in dq.ACTION_STRINGS]
dq.N_ACTIONS = len(dq.ACTION_STRINGS)

MAX_TURNS_DEFAULT = 500
GEM_ORDER = ["diamond", "sapphire", "emerald", "ruby", "onyx"]
BANK_LETTERS = ["D", "S", "E", "R", "O", "G"]


def _bank_summary(bank: dict) -> str:
    key_order = GEM_ORDER + ["gold"]
    parts = []
    for i, g in enumerate(key_order):
        n = bank.get(g, 0)
        parts.append("%s=%d" % (BANK_LETTERS[i], n))
    return "bank:" + " ".join(parts)


def _player_summary(p) -> str:
    pts = p.points
    bon = [p.bonuses.get(g, 0) for g in GEM_ORDER]
    tok = [p.tokens.get(g, 0) for g in GEM_ORDER] + [p.tokens.get("gold", 0)]
    return "pts=%d bon=%s tok=%s" % (pts, bon, tok)


def state_after_line(state: GameState) -> str:
    parts = [_bank_summary(state.bank)]
    for i, p in enumerate(state.players):
        pts = int(getattr(p, "points", 0))
        tot_tok = sum(p.tokens.get(g, 0) for g in GEM_ORDER + ["gold"])
        tot_bon = sum(p.bonuses.get(g, 0) for g in GEM_ORDER)
        parts.append("P%d:pts=%d,tok=%d,cards=%d" % (i, pts, tot_tok, tot_bon))
    parts.append("active=%d" % state.active_idx)
    parts.append("game_over=%s" % getattr(state, "game_over", False))
    if getattr(state, "final_summary", ""):
        parts.append("summary=%s" % state.final_summary[:40])
    return " | ".join(parts)


def _get_buy_card(state: GameState, move) -> "Card|None":
    kind, payload = move
    if kind != "BUY":
        return None
    row, col = payload
    try:
        if row == "R" or row == "r":
            return state.players[state.active_idx].reserved[col]
        table, _, _ = row_to_table_and_deck(state, int(row))
        return table[col]
    except (IndexError, KeyError, TypeError):
        return None


def _card_info(card, include_cost: bool = True) -> str:
    if card is None:
        return ""
    cid = getattr(card, "card_id", None) or getattr(card, "id", "?")
    vp = getattr(card, "victoryPoints", 0)
    gem = getattr(card, "gemType", "?")
    out = " card=%s %dVP %s" % (cid, vp, gem)
    if include_cost and hasattr(card, "cost") and card.cost is not None:
        cost = card.cost
        gem_to_letter = {"Diamond": "D", "Sapphire": "S", "Emerald": "E", "Ruby": "R", "Onyx": "O"}
        cost_parts = []
        for gem, letter in gem_to_letter.items():
            try:
                n = int(cost[gem]) if hasattr(cost, "__getitem__") else 0
            except (KeyError, TypeError):
                n = 0
            if n > 0:
                cost_parts.append("%s:%d" % (letter, n))
        if cost_parts:
            out += " cost=" + " ".join(cost_parts)
    return out


def run_inspect(
    checkpoint_path: str,
    num_games: int = 1,
    trace_dir: str = "traces",
    max_turns: int = MAX_TURNS_DEFAULT,
    seed: int = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_obs = ckpt["n_obs"]
    n_actions = ckpt["n_actions"]
    if n_actions != dq.N_ACTIONS:
        raise ValueError(
            "Checkpoint n_actions=%d does not match current action list length %d"
            % (n_actions, dq.N_ACTIONS)
        )

    policy_net = dq.DQN(n_obs, n_actions).to(device)
    policy_net.load_state_dict(ckpt["policy"])
    policy_net.eval()

    sample_state = load_initial_state(2)
    _ = dq.encode_state(sample_state, turn=0)

    os.makedirs(trace_dir, exist_ok=True)
    hero_idx = 0

    for g in range(num_games):
        state = load_initial_state(2)
        initial_state_copy = copy.deepcopy(state)
        moves_for_replay = []
        results_for_replay = []
        state_summaries = []
        state_snapshots = []

        trace_path = os.path.join(trace_dir, "game_%03d.txt" % (g + 1))
        replay_path = os.path.join(trace_dir, "game_%03d.replay" % (g + 1))
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("checkpoint=%s max_turns=%d game=%d\n" % (checkpoint_path, max_turns, g + 1))
            f.write("Hero=P0 (greedy), P1=random\n")
            f.write("STATE_AFTER each move = canonical state (compare to replay UI)\n")
            f.write("-" * 80 + "\n")

            turn = 0
            while not state.game_over and turn < max_turns:
                active_idx = state.active_idx
                n_legal = dq.legal_action_mask(state).sum()

                bank_str = _bank_summary(state.bank)
                state_line = " ".join(
                    "P%d %s" % (i, _player_summary(state.players[i]))
                    for i in range(len(state.players))
                )

                if active_idx == hero_idx:
                    action_idx = dq.select_greedy_action_with_net(state, policy_net, device, turn=turn)
                else:
                    action_idx = dq.sample_random_legal_action_index(state)

                move = dq.ACTIONS_PARSED[action_idx]
                action_str = dq.ACTION_STRINGS[action_idx] if action_idx < len(dq.ACTION_STRINGS) else str(move)

                buy_card = _get_buy_card(state, move)
                card_suffix = _card_info(buy_card) if buy_card else ""

                result = apply_move(state, move)
                moves_for_replay.append(action_str)
                results_for_replay.append(result.strip())
                state_summaries.append(state_after_line(state))
                state_snapshots.append(copy.deepcopy(state))

                line = (
                    "TURN %4d P%d n_legal=%2d | %s | %s | action=%s%s | %s\n"
                    % (turn, active_idx, n_legal, bank_str, state_line, action_str, card_suffix, result.strip())
                )
                f.write(line)
                f.write("  STATE_AFTER: %s\n" % state_summaries[-1])

                if n_legal == 1 and "kip" in result.lower():
                    f.write("  ^^^ only legal move was skip (possible softlock)\n")

                turn += 1

            f.write("-" * 80 + "\n")
            if state.game_over and state.final_summary:
                f.write("GAME OVER: %s\n" % state.final_summary.strip())
            else:
                f.write("STOPPED: turn cap reached (%d turns)\n" % max_turns)
            scores = [state.players[i].points for i in range(len(state.players))]
            f.write("Final scores: %s | total turns: %d\n" % (scores, turn))

        with open(replay_path, "wb") as rp:
            pickle.dump({
                "initial_state": initial_state_copy,
                "moves": moves_for_replay,
                "results": results_for_replay,
                "state_summaries": state_summaries,
                "state_snapshots": state_snapshots,
                "checkpoint": checkpoint_path,
                "game_index": g + 1,
                "total_turns": turn,
                "game_over": state.game_over,
                "final_scores": [state.players[i].points for i in range(len(state.players))],
            }, rp)
        print("Wrote %s and %s (%d turns, game_over=%s)" % (trace_path, replay_path, turn, state.game_over))

    print("Done. Trace files in %s" % trace_dir)
    print("Replay in UI: python initalize_ui.py <trace_dir>/game_NNN.replay")


def main():
    p = argparse.ArgumentParser(description="Run games with a saved checkpoint and write trace files")
    p.add_argument("checkpoint", help="Path to checkpoint .pt (e.g. checkpoints/hero_ep00050.pt)")
    p.add_argument("--games", type=int, default=1, help="Number of games to run")
    p.add_argument("--trace-dir", default="traces", help="Directory for trace .txt files")
    p.add_argument("--max-turns", type=int, default=MAX_TURNS_DEFAULT, help="Turn cap per game")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = p.parse_args()
    run_inspect(
        args.checkpoint,
        num_games=args.games,
        trace_dir=args.trace_dir,
        max_turns=args.max_turns,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
