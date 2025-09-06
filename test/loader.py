# loader.py
import csv
from card import Card, CardCost
from noble import Noble
from game_state import GameState, Player
import json
import random

def load_cards(path, rank):
    """Load cards from a rankN CSV and tag with rank."""
    cards = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            cost = CardCost(
                onyx=int(row["Cost[Onyx]"]),
                diamond=int(row["Cost[Diamond]"]),
                ruby=int(row["Cost[Ruby]"]),
                sapphire=int(row["Cost[Sapphire]"]),
                emerald=int(row["Cost[Emerald]"]),
            )
            c = Card(
                gemType=row["GemType"],
                victoryPoints=int(row["Points"]),
                cost=cost,
                card_id=f"R{rank}-{i:02d}",
                rank=rank,
            )
            cards.append(c)
    return cards


def load_nobles(path):
    """Load nobles from nobles.csv."""
    nobles = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            n = Noble(
                victoryPoints=int(row["VictoryPoints"]),
                onyx=int(row["Requirement[Onyx]"]),
                diamonds=int(row["Requirement[Diamond]"]),
                rubies=int(row["Requirement[Ruby]"]),
                sapphires=int(row["Requirement[Sapphire]"]),
                emeralds=int(row["Requirement[Emerald]"]),
            )
            nobles.append(n)
    return nobles

def _deal_tableau(deck, n=4):
    table = []
    for _ in range(min(n, len(deck))):
        table.append(deck.pop(0))   # removes from deck, becomes face-up
    return table

def load_initial_state(playerCount: int) -> GameState:
    deck_t1 = load_cards("../data/rank1_cards.csv", rank=1)
    deck_t2 = load_cards("../data/rank2_cards.csv", rank=2)
    deck_t3 = load_cards("../data/rank3_cards.csv", rank=3)
    nobles = load_nobles("../data/nobles.csv")

    random.shuffle(deck_t1); random.shuffle(deck_t2); random.shuffle(deck_t3); random.shuffle(nobles)

    table_t1 = _deal_tableau(deck_t1, n=4)
    table_t2 = _deal_tableau(deck_t2, n=4)
    table_t3 = _deal_tableau(deck_t3, n=4)


    # bank = {
    #     "diamond": 7, "sapphire": 7, "emerald": 7,
    #     "ruby": 7, "onyx": 7, "gold": 5
    # }
    if playerCount in [2, 3]:
        bank = {
            "diamond": 4, "sapphire": 4, "emerald": 4,
            "ruby": 4, "onyx": 4, "gold": 5
        }
    elif playerCount == 4:
        bank = {
            "diamond": 5, "sapphire": 5, "emerald": 5,
            "ruby": 5, "onyx": 5, "gold": 5
        }
    else:
        raise ValueError("playerCount must be 2, 3, or 4")

    # players = [
    #     Player("P1", tokens={c:0 for c in bank}, bonuses={c:0 for c in bank if c!="gold"}),
    #     Player("P2", tokens={c:0 for c in bank}, bonuses={c:0 for c in bank if c!="gold"}),
    #     Player("P3", tokens={c:0 for c in bank}, bonuses={c:0 for c in bank if c!="gold"}),
    #     Player("P4", tokens={c:0 for c in bank}, bonuses={c:0 for c in bank if c!="gold"}),
    # ]

    players = []
    for i in range(playerCount):
        players.append(
            Player(
                name=f"P{i+1}",
                tokens={c:0 for c in bank},
                bonuses={c:0 for c in bank if c!="gold"}
            )
        )

    return GameState(
        bank=bank,
        deck_t1=deck_t1,
        deck_t2=deck_t2,
        deck_t3=deck_t3,
        table_t1=table_t1,
        table_t2=table_t2,
        table_t3=table_t3,
        nobles=nobles,
        players=players,
        active_idx=0
    )
