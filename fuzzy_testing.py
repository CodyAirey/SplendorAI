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
