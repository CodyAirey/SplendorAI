
# fuzzy_testing.py
"""
Simple fuzz tester for the Splendor-like engine, with minimal error printouts.

Can't be bothered with a testing library for now, this works. I think its all I need / want...

Modes:
  - legal : choose only from check_all_available_moves(); any exception is a failure.
  - any   : choose from a superset of syntactically valid moves; ValueError is expected, others fail.

Terminal output:
  - Per-game line showing whether it ended and basic counts.
  - Final summary with averages.
  - Top-K buckets for:
      * Engine rejections (string messages beginning with Invalid/Cannot/Need/No/Unknown)
      * Unexpected Exceptions (type+message)
  - Optional --verbose to print each move/result line.
"""
from __future__ import annotations
import argparse, random, sys, traceback, itertools, re
from typing import List, Tuple, Any, Dict
from collections import Counter

from loader import load_initial_state
from engine import check_all_available_moves, apply_move, GEM_ORDER
from game_state import GameState

ACCEPT_PREFIXES = (
    "Took", "Reserved", "Purchased", "Force Bought", "Turn skipped",
    "Set Coins Successfully.", "GAME OVER"
)
REJECT_PREFIXES = (
    "Invalid", "Cannot", "Need", "No ", "Unknown"
)

def is_accept(msg: str) -> bool:
    return isinstance(msg, str) and msg.startswith(ACCEPT_PREFIXES)

def is_reject(msg: str) -> bool:
    return isinstance(msg, str) and msg.startswith(REJECT_PREFIXES)

def normalize_msg(msg: str, collapse_numbers: bool, collapse_gems: bool) -> str:
    s = msg.strip()
    if collapse_numbers:
        s = re.sub(r"\d+", "#", s)
    if collapse_gems:
        s = re.sub(r"\b(diamond|sapphire|emerald|ruby|onyx|gold)\b", "<gem>", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s

## Move generation

def enumerate_table_moves(state: GameState) -> List[Tuple[str, Any]]:
    moves: List[Tuple[str, Any]] = []
    for row in (0, 1, 2):
        if hasattr(state, "row_to_table_and_deck"):
            table, _, _ = state.row_to_table_and_deck(row)
        else:
            from engine import row_to_table_and_deck
            table, _, _ = row_to_table_and_deck(state, row)
        for col, _c in enumerate(table):
            moves.append(("BUY", (row, col)))
            moves.append(("RESERVE", (row, col)))
    return moves

def enumerate_reserve_buys(state: GameState) -> List[Tuple[str, Any]]:
    moves: List[Tuple[str, Any]] = []
    active = state.players[state.active_idx]
    for col, _c in enumerate(active.reserved):
        moves.append(("BUY", ("R", col)))
    return moves

def enumerate_take_moves(rng: random.Random) -> List[Tuple[str, Any]]:
    moves: List[Tuple[str, Any]] = []
    gems = list(GEM_ORDER)
    rng.shuffle(gems)  # reduce first-gem bias
    for comb in itertools.combinations(gems, 3):
        moves.append(("TAKE_3", tuple(comb)))
    for gem in gems:
        moves.append(("TAKE_2", gem))
    return moves

def all_possible_moves(state: GameState, rng: random.Random) -> List[Tuple[str, Any]]:
    moves = []
    moves.extend(enumerate_table_moves(state))
    moves.extend(enumerate_reserve_buys(state))
    moves.extend(enumerate_take_moves(rng))
    moves.append(("SKIP", None))
    return moves

## Choosers

def choose_legal(state: GameState, rng: random.Random) -> Tuple[str, Any]:
    candidates = list(check_all_available_moves(state))
    return candidates[rng.randrange(len(candidates))]

def choose_any(state: GameState, rng: random.Random) -> Tuple[str, Any]:
    candidates = all_possible_moves(state, rng)
    return candidates[rng.randrange(len(candidates))]

## Core loop

def apply_with_policy(state: GameState, move: Tuple[str, Any], mode: str) -> Tuple[bool, str]:
    try:
        msg = apply_move(state, move)
        return True, msg
    except ValueError as ve:
        if mode == "legal":
            raise
        return False, f"ValueError(expected): {ve}"
    except Exception:
        raise

def run_one_game(players: int, mode: str, rng: random.Random, max_turns: int, verbose: bool,
                 collapse_numbers: bool, collapse_gems: bool,
                 reject_counter: Counter, exc_counter: Counter
                 ) -> Dict[str, Any]:
    state = load_initial_state(players)
    turns = 0
    accepted, rejected = 0, 0
    last_msg = ""
    error = None

    chooser = {"legal": choose_legal, "any": choose_any}[mode]

    try:
        while not getattr(state, "game_over", False) and turns < max_turns:
            turns += 1
            mv = chooser(state, rng)
            try:
                moved, msg = apply_with_policy(state, mv, mode)
            except Exception as e:
                key = f"{type(e).__name__}: {normalize_msg(str(e), collapse_numbers, collapse_gems)}"
                exc_counter[key] += 1
                error = "".join(traceback.format_exception(e))
                break

            last_msg = msg
            if moved and is_accept(msg):
                accepted += 1
            else:
                rejected += 1

            # Count rejections
            if isinstance(msg, str) and is_reject(msg):
                key = normalize_msg(msg, collapse_numbers, collapse_gems)
                reject_counter[key] += 1

            if verbose:
                print(f"[T{turns:04d}] {mv} -> {msg}")
                
    except Exception as e:
        key = f"{type(e).__name__}: {normalize_msg(str(e), collapse_numbers, collapse_gems)}"
        exc_counter[key] += 1
        error = "".join(traceback.format_exception(e))

    return {
        "turns": turns,
        "accepted": accepted,
        "rejected": rejected,
        "game_over": bool(getattr(state, "game_over", False)),
        "last_msg": last_msg,
        "error": error,
        "final_summary": getattr(state, "final_summary", ""),
    }

def print_top(counter: Counter, title: str, top_k: int):
    if not counter:
        print(f"\n{title}: (none)")
        return
    print(f"\n{title}: top {top_k}")
    for i, (k, v) in enumerate(counter.most_common(top_k), 1):
        print(f"  {i:2d}. [{v}] {k}")

def main():
    ap = argparse.ArgumentParser(description="Simple fuzz tester (with minimal error buckets)")
    ap.add_argument("--games", type=int, default=100, help="Number of games to simulate")
    ap.add_argument("--players", type=int, default=4, choices=[2,3,4], help="Players per game")
    ap.add_argument("--max-turns", type=int, default=2000, help="Safety cap on turns per game")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (omit for non-deterministic)")
    ap.add_argument("--mode", choices=["legal", "any"], default="legal")
    ap.add_argument("--collapse-numbers", action="store_true", help="Collapse digits in messages to '#' to reduce bucket fragmentation")
    ap.add_argument("--collapse-gems", action="store_true", help="Collapse gem names in messages to '<gem>'")
    ap.add_argument("--top-k", type=int, default=30, help="How many buckets to show in each top list")
    ap.add_argument("--verbose", action="store_true", help="Print every chosen move/result")
    args = ap.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    totals = {"games": 0, "ended": 0, "errors": 0, "turns": 0, "accepted": 0, "rejected": 0}

    reject_counter: Counter = Counter()
    exc_counter: Counter = Counter()

    for i in range(1, args.games+1):
        res = run_one_game(
            args.players, args.mode, rng, args.max_turns, args.verbose,
            args.collapse_numbers, args.collapse_gems,
            reject_counter, exc_counter
        )
        totals["games"] += 1
        totals["turns"] += res["turns"]
        totals["accepted"] += res["accepted"]
        totals["rejected"] += res["rejected"]
        if res["game_over"]:
            totals["ended"] += 1
        if res["error"]:
            totals["errors"] += 1

        end_note = "ENDED" if res["game_over"] else "NOT ENDED"
        print(f"Game {i:03d}: {end_note} | turns={res['turns']} accepted={res['accepted']} rejected={res['rejected']}")
        if res["final_summary"]:
            print(f"  -> {res['final_summary']}")
        if res["error"]:
            print("  !! ERROR !!")
            print(res["error"])

    print("\n=== FUZZ SUMMARY ===")
    g = totals["games"]
    print(f"Games: {g} | Ended: {totals['ended']} | Errors: {totals['errors']}")
    if g:
        print(f"Avg turns: {totals['turns']/g:.1f} | Avg accepted: {totals['accepted']/g:.1f} | Avg rejected: {totals['rejected']/g:.1f}")

    # Minimal printouts
    print_top(reject_counter, "Engine rejections (string messages)", args.top_k)
    print_top(exc_counter, "Unexpected Exceptions (type+message)", args.top_k)

    if totals["errors"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
