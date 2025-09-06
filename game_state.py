# game_state.py
from dataclasses import dataclass, field
from typing import List, Dict
from card import Card
from noble import Noble

@dataclass
class Player:
    name: str
    tokens: Dict[str, int] = field(default_factory=dict)
    bonuses: Dict[str, int] = field(default_factory=dict)
    points: int = 0
    reserved: List[Card] = field(default_factory=list)

@dataclass
class GameState:
    bank: Dict[str, int]
    deck_t1: List[Card]
    deck_t2: List[Card]
    deck_t3: List[Card]
    table_t1: List[Card]
    table_t2: List[Card]
    table_t3: List[Card]
    nobles: List[Noble]
    players: List[Player]

    start_idx: int = 0
    active_idx: int = 0
    endgame: bool = False              # set when someone first hits ≥15
    endgame_last_player_idx: int = -1  # (start_idx - 1) % n
    game_over: bool = False
    final_summary: str = ""
