# engine.py
from typing import Dict, List, Tuple
from game_state import GameState, Player
from card import Card
import random

# --- helpers -------------------------------------------------------------

LOW = {
    "diamond": "diamond", "sapphire": "sapphire", "emerald": "emerald",
    "ruby": "ruby", "onyx": "onyx", "gold": "gold",
    "Diamond": "diamond", "Sapphire": "sapphire", "Emerald": "emerald",
    "Ruby": "ruby", "Onyx": "onyx", "Gold": "gold",
    "D": "diamond", "S": "sapphire", "E": "emerald", "R": "ruby", "O": "onyx", "G": "gold",
}

GEM_ORDER = ["diamond", "sapphire", "emerald", "ruby", "onyx"]
GEM_ORDER_WITH_GOLD = GEM_ORDER + ["gold"]

def _norm(c: str) -> str:
    return LOW.get(c, c.strip().lower())

def card_cost_lc(card: Card) -> Dict[str, int]:
    # Card.cost is a CardCost exposing items() like ("Ruby", n)
    return {_norm(k): int(v) for k, v in card.cost.items() if int(v) > 0}

def row_to_table_and_deck(state: GameState, row: int) -> Tuple[List[Card], List[Card], int]:
    # UI rows are T3(top)=0, T2=1, T1=2
    if row == 0:   return state.table_t3, state.deck_t3, 3
    if row == 1:   return state.table_t2, state.deck_t2, 2
    if row == 2:   return state.table_t1, state.deck_t1, 1
    raise ValueError("Row must be 0,1,2")

def _effective_need_after_bonuses(card, bonuses: Dict[str, int]) -> Dict[str, int]:
    """Need per color after applying permanent bonuses (no gold here)."""
    eff = {}
    for g, v in card_cost_lc(card).items():  # normalized keys
        if g == "gold":
            continue
        eff[g] = max(0, v - bonuses.get(g, 0))
    # ensure all base colours exist
    for g in GEM_ORDER:
        eff.setdefault(g, 0)
    return eff

def _gold_required_for_card(card, tokens: Dict[str,int], bonuses: Dict[str,int]) -> int:
    """Minimal gold needed to make up remaining deficits after using colored tokens."""
    eff = _effective_need_after_bonuses(card, bonuses)
    need_gold = 0
    for g in GEM_ORDER:
        if g == "gold":
            continue
        deficit = max(0, eff[g] - tokens.get(g, 0))
        need_gold += deficit
    return need_gold

def player_can_afford(p: Player, card: Card) -> bool:
    cost = card_cost_lc(card)
    gold = p.tokens.get("gold", 0)
    deficit = 0
    for gem, need in cost.items():
        need_eff = max(0, need - p.bonuses.get(gem, 0))
        have = p.tokens.get(gem, 0)
        if have < need_eff:
            deficit += (need_eff - have)
    return deficit <= gold

def pay_for_card(p: Player, card: Card) -> Dict[str, int]:
    """Returns dict of tokens to return to bank after payment; mutates player tokens."""
    paid = {"diamond":0,"sapphire":0,"emerald":0,"ruby":0,"onyx":0,"gold":0}
    cost = card_cost_lc(card)

    # First, use colored tokens up to effective need
    for gem, need in cost.items():
        need_eff = max(0, need - p.bonuses.get(gem, 0))
        use = min(p.tokens.get(gem, 0), need_eff)
        if use:
            p.tokens[gem] = p.tokens.get(gem, 0) - use
            paid[gem] += use
        rem = need_eff - use
        if rem > 0:
            # Use gold as wild
            use_gold = min(p.tokens.get("gold", 0), rem)
            p.tokens["gold"] = p.tokens.get("gold", 0) - use_gold
            paid["gold"] += use_gold
    return paid

def apply_purchase(state: GameState, row: int, col: int) -> str:
    p = state.players[state.active_idx]
    table, deck, _tier = row_to_table_and_deck(state, row)
    if not (0 <= col < len(table)):
        return "Invalid BUY: card position not on table."

    card = table[col]
    if not player_can_afford(p, card):
        return "Cannot afford that card."

    paid = pay_for_card(p, card)
    # Return payment to bank
    for g, n in paid.items():
        if n: state.bank[g] = state.bank.get(g, 0) + n

    # Gain bonus and points
    bonus_gem = _norm(card.gemType)
    if bonus_gem != "gold":
        p.bonuses[bonus_gem] = p.bonuses.get(bonus_gem, 0) + 1
    p.points += int(card.victoryPoints)

    # Remove from table; refill from deck if possible
    table.pop(col)
    if deck:
        table.insert(col, deck.pop(0))

    # Advance turn
    state.active_idx = (state.active_idx + 1) % len(state.players)
    return "Purchased."

def apply_reserve(state: GameState, row: int, col: int) -> str:
    p = state.players[state.active_idx]
    if len(p.reserved) >= 3:
        return "You already have 3 reserved cards."

    table, deck, _tier = row_to_table_and_deck(state, row)
    if not (0 <= col < len(table)):
        return "Invalid RESERVE: card position not on table."

    # Take gold if available
    if state.bank.get("gold", 0) > 0:
        state.bank["gold"] -= 1
        p.tokens["gold"] = p.tokens.get("gold", 0) + 1

    # Move card from table to player's reserved; refill
    card = table.pop(col)
    p.reserved.append(card)
    if deck:
        table.insert(col, deck.pop(0))

    # Advance turn
    state.active_idx = (state.active_idx + 1) % len(state.players)
    return "Reserved."

def apply_take2(state: GameState, gem: str) -> str:
    g = _norm(gem)
    if g == "gold": return "Cannot take gold with TAKE action."
    if state.bank.get(g, 0) < 4:
        return f"Need ≥4 {g} in bank to take 2; only {state.bank.get(g,0)} available."
    # Take exactly 2
    state.bank[g] -= 2
    p = state.players[state.active_idx]
    p.tokens[g] = p.tokens.get(g, 0) + 2

    state.active_idx = (state.active_idx + 1) % len(state.players)
    return f"Took 2 {g}."


def apply_take3(state: GameState, gems: Tuple[str, str, str]) -> str:
    gs = [_norm(x) for x in gems]
    if len(set(gs)) != 3 or "gold" in gs:
        return "TAKE_3 must be 3 distinct non-gold colours."

    # availability
    for g in gs:
        if state.bank.get(g, 0) <= 0:
            return f"No {g} left in bank."

    # take 1 each
    p = state.players[state.active_idx]
    for g in gs:
        state.bank[g] -= 1
        p.tokens[g] = p.tokens.get(g, 0) + 1

    # 10-token cap (gold counts), but we will never discard gold
    total = sum(p.tokens.get(x, 0) for x in GEM_ORDER_WITH_GOLD)
    if total <= 10:
        state.active_idx = (state.active_idx + 1) % len(state.players)
        return f"Took {', '.join(gs)}."



    # heuristic discard: try to keep tokens needed for purchasable cards
    # TODO: move this logic out for other actions that can cause >10
    over = total - 10
    returned = {g: 0 for g in GEM_ORDER}  # track only coloured returns

    # --- build purchasable set (table only); sort by best value proposition
    purchasable = []
    #add tabe cards
    for row in (0, 1, 2):
        table, _, _ = row_to_table_and_deck(state, row)
        for c in table:
            if player_can_afford(p, c):
                value_prop = c.getCostPerPoint() if hasattr(c, "getCostPerPoint") else (
                    (sum(card_cost_lc(c).values()) / max(1, int(getattr(c, "victoryPoints", 0))))
                    if int(getattr(c, "victoryPoints", 0)) > 0 else float("inf")
                )
                purchasable.append((value_prop, c))
    #add reserved cards
    for c in state.players[state.active_idx].reserved:
        if player_can_afford(p, c):
                value_prop = c.getCostPerPoint() if hasattr(c, "getCostPerPoint") else (
                    (sum(card_cost_lc(c).values()) / max(1, int(getattr(c, "victoryPoints", 0))))
                    if int(getattr(c, "victoryPoints", 0)) > 0 else float("inf")
                )
                purchasable.append((value_prop, c))
    purchasable.sort(key=lambda t: t[0])

    # --- phase 1: discard only "surplus" vs. each purchasable card 
    for _, card in purchasable:
        if over <= 0:
            break

        eff_need = _effective_need_after_bonuses(card, p.bonuses)
        gold_req = _gold_required_for_card(card, p.tokens, p.bonuses)  # gold needed to cover remaining deficits

        # Minimal spend to afford this card:
        min_spend = {g: 0 for g in GEM_ORDER_WITH_GOLD}
        for g in GEM_ORDER:
            min_spend[g] = min(p.tokens.get(g, 0), eff_need[g])
        min_spend["gold"] = min(p.tokens.get("gold", 0), gold_req)

        # Surplus per COLOURED gem = tokens - min_spend (never consider gold for discard)
        surplus = {g: max(0, p.tokens.get(g, 0) - min_spend[g]) for g in GEM_ORDER}

        order = GEM_ORDER[:]     # coloured only
        random.shuffle(order)
        for g in order:
            if over <= 0:
                break
            can_put = min(surplus[g], over)
            if can_put > 0:
                p.tokens[g] -= can_put
                state.bank[g] = state.bank.get(g, 0) + can_put
                returned[g] += can_put
                over -= can_put

    # --- phase 2 (fallback): still over → shave largest COLOURED piles
    if over > 0:
        piles = sorted(GEM_ORDER, key=lambda g: p.tokens.get(g, 0), reverse=True)
        for g in piles:
            if over <= 0:
                break
            have = p.tokens.get(g, 0)
            if have <= 0:
                continue
            cnt = min(have, over)
            p.tokens[g] -= cnt
            state.bank[g] = state.bank.get(g, 0) + cnt
            returned[g] += cnt
            over -= cnt

    # Advance turn & report
    state.active_idx = (state.active_idx + 1) % len(state.players)
    discards_msg = ", ".join(f"{g}:{n}" for g, n in returned.items() if n > 0)
    return f"Took {', '.join(gs)}." + (f" Discarded ({discards_msg})." if discards_msg else "")


# --- public API ----------------------------------------------------------

def apply_move(state: GameState, parsed_move) -> str:
    """
    parsed_move comes from your parse_move(), e.g.:
      ("TAKE_2", "D")
      ("TAKE_3", ("D","S","E"))
      ("BUY", (row, col))
      ("RESERVE", (row, col))
    Returns a short status string; mutates state in place.
    """
    kind, payload = parsed_move

    if kind == "TAKE_2":
        return apply_take2(state, payload)

    if kind == "TAKE_3":
        # payload is a 3-tuple of gem letters/names
        return apply_take3(state, payload)

    if kind == "RESERVE":
        row, col = payload
        return apply_reserve(state, int(row), int(col))

    if kind == "BUY":
        row, col = payload
        #TODO handle buying from reserve
        return apply_purchase(state, int(row), int(col))
    
    if kind == "SKIP":
        state.active_idx = (state.active_idx + 1) % len(state.players)
        return "Turn skipped."

    return "Unknown move."
