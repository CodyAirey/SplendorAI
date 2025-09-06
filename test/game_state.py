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
    active_idx: int = 0
