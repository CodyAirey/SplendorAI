# engine.py
"""
Splendor rules/engine layer:
- Random starting player
- Four actions (take3, take2, reserve, buy)
- Validates + applies, refills tableau, rotates turns
- Parser-friendly adapter (apply_move_str)
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional, Iterable

import move_parser


# ---- Colour helpers ---------------------------------------------------------

COLORS_NO_GOLD = ["diamond", "sapphire", "emerald", "ruby", "onyx"]
ALL_COLORS     = COLORS_NO_GOLD + ["gold"]
LETTER_TO_COLOR = {
    "D": "diamond", "S": "sapphire", "E": "emerald", "R": "ruby", "O": "onyx", "G": "gold"
}


def _norm_colour(s: str) -> str:
    return s.strip().lower()


# ---- Engine ----------------------------------------------------------------

class SplendorEngine:
    """
    Stateless-ish executor operating on a mutable GameState (passed in or held by ref).
    Expects GameState to have:
      - bank: Dict[str,int] with keys lowercased colours (diamond,...,gold)
      - deck_t1, deck_t2, deck_t3: List[Card]
      - table_t1, table_t2, table_t3: List[Card] (face-up)
      - nobles: List[Noble]   (not used here yet)
      - players: List[Player] with fields: tokens:Dict, bonuses:Dict, points:int, reserved:List[Card]
      - active_idx: int
    Card model with fields: gemType (str), victoryPoints (int), cost (CardCost), rank (1..3)
    """

    def __init__(self, state):
        self.state = state
        # Map row index (parser/UI) to (table, deck). Row 0=Tier3, 1=Tier2, 2=Tier1 (matches your UI)
        self._row_map = {
            0: ("table_t3", "deck_t3", 3),
            1: ("table_t2", "deck_t2", 2),
            2: ("table_t1", "deck_t1", 1),
        }

    # -------------------- Start / Turn --------------------------------------

    def start_game(self, seed: Optional[int] = None) -> None:
        """Pick a random starting player."""
        if seed is not None:
            rnd = random.Random(seed)
            self.state.active_idx = rnd.randrange(len(self.state.players))
        else:
            self.state.active_idx = random.randrange(len(self.state.players))

    def _next_player(self) -> None:
        self.state.active_idx = (self.state.active_idx + 1) % len(self.state.players)

    # -------------------- Public “apply” API (structured) -------------------

    def take_three(self, colours: Iterable[str]) -> Tuple[bool, str]:
        """Take 3 distinct non-gold colours (each pile must have at least 1)."""
        cols = [ _norm_colour(c) for c in colours ]
        if len(cols) != 3 or len(set(cols)) != 3:
            return False, "Pick exactly 3 distinct colours."

        if any(c not in COLORS_NO_GOLD for c in cols):
            return False, "Gold cannot be taken in Take-3, and colours must be valid."

        bank = self.state.bank
        if any(bank.get(c, 0) <= 0 for c in cols):
            return False, "One or more selected piles are empty."

        # Apply
        p = self.state.players[self.state.active_idx]
        for c in cols:
            bank[c] -= 1
            p.tokens[c] = p.tokens.get(c, 0) + 1

        self._next_player()
        return True, "Took 3 different tokens."

    def take_two(self, colour: str) -> Tuple[bool, str]:
        """Take 2 of the same colour (only if that pile has >= 4 before taking)."""
        c = _norm_colour(colour)
        if c not in COLORS_NO_GOLD:
            return False, "Cannot take gold or an unknown colour for Take-2."

        bank = self.state.bank
        if bank.get(c, 0) < 4:
            return False, "Pile must have at least 4 tokens before taking 2."

        # Apply
        p = self.state.players[self.state.active_idx]
        bank[c] -= 2
        p.tokens[c] = p.tokens.get(c, 0) + 2

        self._next_player()
        return True, "Took 2 of the same colour."

    def reserve_from_table(self, row: int, col: int) -> Tuple[bool, str]:
        """
        Reserve one face-up card from the tableau (row: 0=Tier3,1=Tier2,2=Tier1).
        - Move card to player's reserve (max 3).
        - Take 1 gold if available.
        - Refill the tableau slot from the corresponding deck if possible; else the row shrinks.
        """
        ok, msg, table, deck, _tier = self._get_table_ref(row)
        if not ok:
            return False, msg

        if not (0 <= col < len(table)):
            return False, "That card slot is empty."

        p = self.state.players[self.state.active_idx]
        if len(p.reserved) >= 3:
            return False, "Reserve limit reached (3)."

        card = table[col]

        # Move to reserve
        p.reserved.append(card)

        # Give gold if available
        if self.state.bank.get("gold", 0) > 0:
            self.state.bank["gold"] -= 1
            p.tokens["gold"] = p.tokens.get("gold", 0) + 1

        # Refill that face-up slot
        self._refill_slot(table, deck, col)

        self._next_player()
        return True, "Reserved the card."

    def buy_from_table(self, row: int, col: int) -> Tuple[bool, str]:
        ok, msg, table, deck, _tier = self._get_table_ref(row)
        if not ok:
            return False, msg
        if not (0 <= col < len(table)):
            return False, "That card slot is empty."
        return self._buy_card(table, deck, col)

    def buy_reserved(self, reserve_index: int) -> Tuple[bool, str]:
        p = self.state.players[self.state.active_idx]
        if not (0 <= reserve_index < len(p.reserved)):
            return False, "No such reserved card."
        card = p.reserved[reserve_index]
        # Try to pay
        ok, msg, payment = self._compute_payment(p, card)
        if not ok:
            return False, msg

        self._apply_payment(p, payment)
        self._acquire_card(p, card)
        # Remove from reserves
        p.reserved.pop(reserve_index)

        self._next_player()
        return True, "Purchased reserved card."

    # -------------------- Parser adapter (optional) -------------------------

    def apply_move_str(self, move_str: str) -> Tuple[bool, str]:
        """
        Accepts strings like:
          T(DSE)       -> take_three
          T(DD)        -> take_two
          R(1,3)       -> reserve_from_table(row=1,col=3)
          B(2,0)       -> buy_from_table(row=2,col=0)
          B(R,1)       -> buy_reserved(index=1)
        """
        kind, payload = None, None
        if move_parser:
            try:
                kind, payload = move_parser.parse_move(move_str)
            except Exception as e:
                return False, f"Invalid move format: {e}"
        else:
            # Minimal fallback if parser module not available
            return False, "Parser module not found; use structured methods."

        # ---- TAKE 2 or TAKE 3
        if kind == "TAKE_2":
            gem_letter = payload  # e.g., "D"
            colour = LETTER_TO_COLOR.get(gem_letter)
            if not colour:
                return False, "Unknown gem letter."
            return self.take_two(colour)

        if kind == "TAKE_3":
            letters = payload  # tuple like ("D","S","E")
            colours = []
            for L in letters:
                c = LETTER_TO_COLOR.get(L)
                if not c or c == "gold":
                    return False, "Invalid colours for Take-3."
                colours.append(c)
            return self.take_three(colours)

        # ---- BUY
        if kind == "BUY":
            row, col = payload
            # Support B(R,idx) for buying from reserve
            if isinstance(row, str) and row.upper() == "R":
                try:
                    idx = int(col)
                except Exception:
                    return False, "Invalid reserve index."
                return self.buy_reserved(idx)
            # Normal table buy
            try:
                r = int(row); c = int(col)
            except Exception:
                return False, "Row/col must be integers (or R,<idx> for reserved)."
            return self.buy_from_table(r, c)

        # ---- RESERVE
        if kind == "RESERVE":
            row, col = payload
            try:
                r = int(row); c = int(col)
            except Exception:
                return False, "Row/col must be integers."
            return self.reserve_from_table(r, c)

        return False, "Unknown move."

    # -------------------- Internals -----------------------------------------

    def _get_table_ref(self, row: int):
        if row not in self._row_map:
            return False, "Row must be 0 (T3), 1 (T2), or 2 (T1).", None, None, None
        table_name, deck_name, tier = self._row_map[row]
        table = getattr(self.state, table_name)
        deck  = getattr(self.state, deck_name)
        return True, "", table, deck, tier

    def _refill_slot(self, table: List, deck: List, idx: int) -> None:
        """Refill a specific tableau index from deck, else shrink the row."""
        if deck:
            # Replace the same slot
            table[idx] = deck.pop(0)
        else:
            # Remove the slot entirely
            table.pop(idx)

    def _buy_card(self, table: List, deck: List, col: int) -> Tuple[bool, str]:
        p = self.state.players[self.state.active_idx]
        card = table[col]

        ok, msg, payment = self._compute_payment(p, card)
        if not ok:
            return False, msg

        self._apply_payment(p, payment)
        self._acquire_card(p, card)

        # Refill that face-up slot
        self._refill_slot(table, deck, col)

        self._next_player()
        return True, "Purchased."

    def _compute_payment(self, player, card) -> Tuple[bool, str, Dict[str,int]]:
        """
        Compute how many coloured tokens + gold are needed after bonuses.
        Returns (ok, msg, payment_dict[all colours]).
        """
        # Build required after bonuses
        need_by = {c: 0 for c in ALL_COLORS}
        # Card cost keys are title-cased; normalise to lowercase
        for k, v in card.cost.items():
            c = _norm_colour(k)
            if v <= 0: 
                continue
            base = int(v)
            bonus = int(player.bonuses.get(c, 0))
            need_by[c] = max(0, base - bonus)

        # Spend coloured tokens first
        pay = {c: 0 for c in ALL_COLORS}
        remaining = 0
        for c in COLORS_NO_GOLD:
            use = min(player.tokens.get(c, 0), need_by[c])
            pay[c] = use
            remaining += (need_by[c] - use)

        # Fill remainder with gold
        gold_avail = player.tokens.get("gold", 0)
        if remaining > gold_avail:
            return False, "Cannot afford (not enough gold).", {}
        pay["gold"] = remaining
        return True, "OK", pay

    def _apply_payment(self, player, payment: Dict[str,int]) -> None:
        """Return tokens to bank based on computed payment."""
        for c, n in payment.items():
            if not n:
                continue
            player.tokens[c] -= n
            self.state.bank[c] += n

    def _acquire_card(self, player, card) -> None:
        """Grant the card's bonus and points to the player."""
        bonus_colour = _norm_colour(card.gemType)
        if bonus_colour in COLORS_NO_GOLD:
            player.bonuses[bonus_colour] = player.bonuses.get(bonus_colour, 0) + 1
        player.points += int(card.victoryPoints)
