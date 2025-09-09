import os
from loader import load_possible_moves

# Cheat codes
FORCE_BUY = "FB("
SKIP = "SKIP"
SET_COINS = "SC("

TAKE = "T("
BUY = "B("
RESERVE = "R("

END = ")"

RUBY = "R"
ONYX = "O"
SAPPHIRE = "S"
DIAMOND = "D"
EMERALD = "E"
GOLD = "G"

TABLE_ROWS = 3   # tiers: 0..2
TABLE_COLS = 4   # columns: 0..3
RESERVE_COLS = 3 # per-player reserve slots: 0..2

GEMS = [DIAMOND, SAPPHIRE, EMERALD, RUBY, ONYX, GOLD]
GEM_SET = set(GEMS)
GEM_SET_NO_GOLD = set(GEMS) - {GOLD}


def parse_move(move_str):
    move_str = move_str.strip().upper()

    if move_str == SKIP:
        return ("SKIP", None)
    
    # ---- SET_COINS ----
    if move_str.startswith(SET_COINS) and move_str.endswith(END):
        body = move_str[len(SET_COINS):-len(END)].replace(" ", "")
        if "," in body:
            parts = body.split(",")
        else:
            parts = list(body)

        if len(parts) != 6:
            raise ValueError("SET_COINS needs 6 values: D,S,E,R,O,G.")

        try:
            _ = [int(x) for x in parts]
            return ("SET_COINS", parts)
        except ValueError:
            raise ValueError("SET_COINS expects digits only (D,S,E,R,O,G).")


    # ---- FORCE_BUY ----
    if move_str.startswith(FORCE_BUY) and move_str.endswith(END):
        params = move_str[len(FORCE_BUY):-len(END)].split(",")
        if len(params) != 2:
            raise ValueError(f"Invalid FORCE_BUY params: {move_str}")
        a, b = params[0].strip(), params[1].strip()
        if a == "R":
            return ("FORCE_BUY", ("R", int(b)))
        return ("FORCE_BUY", (int(a), int(b)))

    # ---- TAKE ----
    if move_str.startswith(TAKE) and move_str.endswith(END):
        gems = move_str[len(TAKE):-len(END)]
        # two of the same, non-gold: e.g. "T(DD)"
        if len(gems) == 2 and gems[0] == gems[1] and gems[0] in GEM_SET_NO_GOLD:
            return ("TAKE_2", gems[0])
        # three distinct, non-gold: e.g. "T(DSE)"
        if len(gems) == 3 and all(g in GEM_SET_NO_GOLD for g in gems) and len(set(gems)) == 3:
            return ("TAKE_3", tuple(sorted(gems)))
        raise ValueError(f"Invalid TAKE move: {move_str}")

    # ---- BUY ----
    if move_str.startswith(BUY) and move_str.endswith(END):
        params = move_str[len(BUY):-len(END)].split(",")
        if len(params) != 2:
            raise ValueError(f"Invalid BUY move parameters: {move_str}")
        a, b = params[0].strip(), params[1].strip()

        if a == "R":
            idx = int(b)
            if not (0 <= idx < RESERVE_COLS):
                raise ValueError(f"BUY reserve index out of range: {idx}")
            return ("BUY", ("R", idx))

        # Only table buys (row,col) supported for now
        try:
            row = int(a)
            col = int(b)
        except ValueError:
            raise ValueError(f"Invalid BUY move indices: {move_str}")

        if not (0 <= row < TABLE_ROWS) or not (0 <= col < TABLE_COLS):
            raise ValueError(f"BUY out of range: row {row} col {col}")
        return ("BUY", (row, col))

    # ---- RESERVE ----
    if move_str.startswith(RESERVE) and move_str.endswith(END):
        params = move_str[len(RESERVE):-len(END)].split(",")
        if len(params) != 2:
            raise ValueError(f"Invalid RESERVE move parameters: {move_str}")
        try:
            row = int(params[0])
            col = int(params[1])
        except ValueError:
            raise ValueError(f"Invalid RESERVE move indices: {move_str}")

        if not (0 <= row < TABLE_ROWS) or not (0 <= col < TABLE_COLS):
            raise ValueError(f"RESERVE out of range: row {row} col {col}")
        return ("RESERVE", (row, col))

    # ---- Unknown ----
    raise ValueError(f"Unknown move format: {move_str}")


# ----------------------------- tests -----------------------------

def _run_case(move, expect_kind=None, expect_value=None, should_fail=False):
    try:
        got = parse_move(move)
        if should_fail:
            print(f"FAIL  {move:<10}  expected error, got {got}")
            return 1
        if expect_kind is not None and got[0] != expect_kind:
            print(f"FAIL  {move:<10}  kind {got[0]} != {expect_kind}")
            return 1
        if expect_value is not None and got[1] != expect_value:
            print(f"FAIL  {move:<10}  value {got[1]} != {expect_value}")
            return 1
        print(f"PASS  {move:<10}  -> {got}")
        return 0
    except Exception as e:
        if should_fail:
            print(f"PASS  {move:<10}  raised {type(e).__name__}: {e}")
            return 0
        else:
            print(f"FAIL  {move:<10}  unexpected {type(e).__name__}: {e}")
            return 1

def test_parser_with_file():
    failures = 0

    valid_moves = load_possible_moves()

    print(f"Number of valid moves: {len(valid_moves)}")

    print("=== Valid moves from file ===")
    for move in valid_moves:
        failures += _run_case(move, should_fail=False)

    # --- Some intentionally invalid moves ---
    invalid_moves = [
        "T(2D)",   # wrong syntax
        "T(D)",    # too short
        "T(DDG)",  # gold not allowed in take
        "T(DR)",   # must be 2 of same
        "T(DDR)",  # cant take 2 of a gem + 1 other
        "B(3,0)",  # row out of range
        "B(0,4)",  # col out of range
        "B(R,3)",  # reserve col out of range
        "R(3,0)",  # row out of range
        "R(0,4)",  # col out of range
        "X(DSR)",  # unknown action
        "2(DSR)",  # unknown action
    ]

    print("\n=== Invalid moves (expected to fail) ===")
    for move in invalid_moves:
        failures += _run_case(move, should_fail=True)

    # --- Summary ---
    print("\n=== Summary ===")
    if failures:
        print(f"{failures} test(s) failed.")
    else:
        print("All tests passed.")


if __name__ == "__main__":
    test_parser_with_file()
