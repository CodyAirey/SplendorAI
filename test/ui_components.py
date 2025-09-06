# splendor_template.py
# Splendor-style board scaffold in Pygame with centred nobles, decks-left,
# centred card grid, bottom-centred coin row, dynamic player panel,
# and a right-side "Reserves" column where reserved cards appear and are clickable.

import pygame
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random

pygame.init()
pygame.display.set_caption("Splendor Template")

# ------------------------ Config & Constants ------------------------

BASE_W, H = 1280, 780     # board area width + player column (no reserves)
PANEL_PLAYERS_W = 360     # left part of right-side panel: player cards
RESERVE_W = 420           # extra room to the right for the "Reserves" column
W = BASE_W + RESERVE_W    # total window width so board stays the same size

FPS = 60
SCREEN = pygame.display.set_mode((W, H))
CLOCK = pygame.time.Clock()

# Colours
BLACK = (0, 0, 0)
WHITE = (240, 240, 240)
GREY = (200, 200, 200)
DARKGREY = (60, 60, 60)
GOLD = (255, 215, 0)

# Token colour names (standard Splendor colours + gold/joker)
TOKEN_COLOURS = ["diamond", "sapphire", "emerald", "ruby", "onyx", "gold"]
COLOUR_TO_RGB = {
    "diamond": (220, 220, 255),
    "sapphire": (70, 110, 255),
    "emerald": (70, 180, 120),
    "ruby": (220, 60, 60),
    "onyx": (40, 40, 40),
    "gold": (240, 210, 60),
}

# Layout (pixels)
MARGIN = 16
PANEL_W = PANEL_PLAYERS_W + RESERVE_W      # total right-side panel width
BOARD_LEFT = MARGIN
BOARD_RIGHT = W - PANEL_W - 2*MARGIN
BOARD_W = BOARD_RIGHT - BOARD_LEFT
ROW_H = 160                 # card row height
CARD_W, CARD_H = 110, 150
NOBLE_W, NOBLE_H = 110, 110
TOKEN_R = 36                # radius for token chips
GAP_X = 20                  # horizontal gap between tiles/cards
GAP_Y = 12                  # vertical row gap
STATUS_H = 44               # reserved strip at bottom for the status text

FONT = pygame.font.SysFont("arial", 18)
FONT_SMALL = pygame.font.SysFont("arial", 14)
FONT_BIG = pygame.font.SysFont("arial", 22, bold=True)


# --- Model → UI normalisers (support both your real models and the old mock) ---

def _norm_colour(name: str) -> str:
    return name.strip().lower()

def ui_card_points(card) -> int:
    return getattr(card, "points", getattr(card, "victoryPoints", 0))

def ui_card_bonus(card) -> str:
    raw = getattr(card, "bonus", getattr(card, "gemType", ""))
    return _norm_colour(raw)

def ui_card_cost(card) -> dict:
    # mock: dict; real: CardCost with .items()
    if hasattr(card, "cost"):
        if isinstance(card.cost, dict):
            src = card.cost.items()
        elif hasattr(card.cost, "items"):
            src = card.cost.items()  # CardCost.items() yields ("Ruby", n), etc.
        else:
            src = []
    else:
        src = []
    return { _norm_colour(k): int(v) for k, v in src if int(v) > 0 }

def ui_noble_points(noble) -> int:
    return getattr(noble, "points", getattr(noble, "victoryPoints", 0))

def ui_noble_requirements(noble) -> dict:
    if hasattr(noble, "requirement"):   # old mock shape
        src = noble.requirement.items()
    else:  # your real Noble shape
        src = [
            ("Emerald", getattr(noble, "emeralds", 0)),
            ("Diamond", getattr(noble, "diamonds", 0)),
            ("Sapphire", getattr(noble, "sapphires", 0)),
            ("Onyx", getattr(noble, "onyx", 0)),
            ("Ruby", getattr(noble, "rubies", 0)),
        ]
    return { _norm_colour(k): int(v) for k, v in src if int(v) > 0 }

# ------------------------ Data Models ------------------------

@dataclass
class Card:
    tier: int                     # 1, 2, 3
    points: int
    bonus: str                    # one of TOKEN_COLOURS w/o gold
    cost: Dict[str, int]          # e.g., {"diamond": 1, "ruby": 2}
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0,0,CARD_W,CARD_H))

@dataclass
class Noble:
    points: int
    requirement: Dict[str, int]   # bonus requirements
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0,0,NOBLE_W,NOBLE_H))

@dataclass
class Player:
    name: str
    tokens: Dict[str, int] = field(default_factory=lambda: {c:0 for c in TOKEN_COLOURS})
    bonuses: Dict[str, int] = field(default_factory=lambda: {c:0 for c in TOKEN_COLOURS if c!="gold"})
    points: int = 0
    reserved: List[Card] = field(default_factory=list)

@dataclass
class TokenBank:
    counts: Dict[str, int]

# ------------------------ Rendering ------------------------

def draw_token_chip(surface, centre: Tuple[int,int], colour: str, label: str):
    pygame.draw.circle(surface, COLOUR_TO_RGB[colour], centre, TOKEN_R)
    pygame.draw.circle(surface, DARKGREY, centre, TOKEN_R, 3)
    txt = FONT_BIG.render(label, True, BLACK if colour!="onyx" else WHITE)
    surface.blit(txt, txt.get_rect(center=centre))

def draw_card(surface, card):
    # rect already assigned elsewhere
    pygame.draw.rect(surface, WHITE, card.rect, border_radius=8)
    pygame.draw.rect(surface, DARKGREY, card.rect, 2, border_radius=8)

    bonus = ui_card_bonus(card)
    points = ui_card_points(card)
    cost = ui_card_cost(card)

    # Header strip by bonus colour
    header = pygame.Rect(card.rect.x, card.rect.y, card.rect.w, 28)
    pygame.draw.rect(surface, COLOUR_TO_RGB.get(bonus, GREY), header, border_radius=8)

    # Points
    pts = FONT_BIG.render(str(points) if points else "", True, GOLD)
    surface.blit(pts, (card.rect.x+8, card.rect.y))

    # Tier (colour-aware text)
    tier_val = getattr(card, "tier", getattr(card, "rank", "?"))
    tier_colour = WHITE if bonus == "onyx" else BLACK
    tier = FONT_SMALL.render(f"T{tier_val}", True, tier_colour)
    surface.blit(tier, (card.rect.right-28, card.rect.y+6))

    # Cost (bottom-right stacked)
    y = card.rect.y + 126
    for c, v in sorted(cost.items()):
        dot = pygame.Surface((14,14), pygame.SRCALPHA)
        pygame.draw.circle(dot, COLOUR_TO_RGB.get(c, GREY), (7,7), 7)
        surface.blit(dot, (card.rect.x+10, y+2))
        cost_txt = FONT_SMALL.render(str(v), True, BLACK)
        surface.blit(cost_txt, (card.rect.x+28, y-2))
        y -= 18


def draw_noble(surface, noble):
    pygame.draw.rect(surface, (245, 235, 210), noble.rect, border_radius=8)
    pygame.draw.rect(surface, DARKGREY, noble.rect, 2, border_radius=8)

    pts_val = ui_noble_points(noble)
    req = ui_noble_requirements(noble)

    label = FONT_SMALL.render("Noble", True, BLACK)
    surface.blit(label, (noble.rect.x + 68, noble.rect.y + 8))

    pts = FONT_BIG.render(str(pts_val), True, BLACK)
    surface.blit(pts, (noble.rect.x+8, noble.rect.y+8))

    y = noble.rect.y + 90
    for c, v in sorted(req.items()):
        dot = pygame.Surface((12,12), pygame.SRCALPHA)
        pygame.draw.circle(dot, COLOUR_TO_RGB.get(c, GREY), (6,6), 6)
        surface.blit(dot, (noble.rect.x+10, y))
        req_txt = FONT_SMALL.render(str(v), True, BLACK)
        surface.blit(req_txt, (noble.rect.x+26, y-2))
        y -= 16


def draw_deck_pile(surface, rect: pygame.Rect, tier: int, remaining: int):
    pygame.draw.rect(surface, (210, 200, 170), rect, border_radius=8)
    pygame.draw.rect(surface, DARKGREY, rect, 2, border_radius=8)
    surface.blit(FONT_SMALL.render(f"T{tier}", True, BLACK), (rect.x+8, rect.y+6))
    txt = FONT_BIG.render(str(remaining), True, BLACK)
    surface.blit(txt, txt.get_rect(center=(rect.centerx, rect.centery+8)))

def draw_player_panel(surface, players: List[Player], active_idx: int, x0: int, y0: int):
    # Entire right panel background
    panel_rect = pygame.Rect(x0, 0, PANEL_W, H)
    pygame.draw.rect(surface, (250, 250, 255), panel_rect)

    # Column splits
    players_x = x0
    reserves_x = x0 + PANEL_PLAYERS_W

    # Headers
    hdr = FONT_BIG.render("Players", True, BLACK)
    surface.blit(hdr, (players_x + 12, y0))
    hdr_r = FONT_BIG.render("Reserves", True, BLACK)
    surface.blit(hdr_r, (reserves_x + 12, y0))

    # Players column
    n = min(4, len(players))
    top = y0 + 30
    block_h = 150   # taller to fit stacked rows + total line
    pad = 16

    for idx in range(n):
        p = players[idx]
        # Player card block (left sub-column)
        block = pygame.Rect(players_x + 12, top + idx*(block_h + pad), PANEL_PLAYERS_W - 24, block_h)
        pygame.draw.rect(surface, WHITE, block, border_radius=8)
        pygame.draw.rect(surface, DARKGREY, block, 2, border_radius=8)

        name = f"{'> ' if idx==active_idx else ''}{p.name}  (Pts {p.points})"
        surface.blit(FONT.render(name, True, BLACK), (block.x+12, block.y+8))

        # ---- Tokens + bonuses stacked by colour ----
        available = block.w - 24
        step = max(40, available // len(TOKEN_COLOURS))
        start_x = block.x + 12 + step // 2

        cy_tokens = block.y + 50
        cy_bonus  = cy_tokens + 28

        for i, c in enumerate(TOKEN_COLOURS):
            cx = start_x + i * step

            # token circle
            pygame.draw.circle(surface, COLOUR_TO_RGB[c], (cx, cy_tokens), 10)
            pygame.draw.circle(surface, DARKGREY, (cx, cy_tokens), 10, 1)
            cnt = str(p.tokens.get(c, 0))
            txt = FONT_SMALL.render(cnt, True, WHITE if c == "onyx" else BLACK)
            surface.blit(txt, txt.get_rect(center=(cx, cy_tokens)))

            # bonus square (skip gold)
            if c != "gold":
                size = 18
                rect = pygame.Rect(cx - size//2, cy_bonus - size//2, size, size)
                pygame.draw.rect(surface, COLOUR_TO_RGB[c], rect, border_radius=4)
                pygame.draw.rect(surface, DARKGREY, rect, 1, border_radius=4)
                b = str(p.bonuses.get(c, 0))
                txt = FONT_SMALL.render(b, True, BLACK if c != "onyx" else WHITE)
                surface.blit(txt, txt.get_rect(center=rect.center))

        # ---- Total coin count ----
        total = sum(p.tokens.values())
        cy_total = cy_bonus + 26
        total_txt = FONT_SMALL.render(f"Total coins: {total}", True, BLACK)
        surface.blit(total_txt, total_txt.get_rect(center=(block.centerx, cy_total)))

        # ---- Reserves row (right sub-column), clickable cards ----
        row_top = top + idx*(block_h + pad)
        row_y   = row_top + (block_h - CARD_H)//2 + 2

        start_rx = reserves_x + 12
        for j, c in enumerate(p.reserved[:3]):
            r = pygame.Rect(start_rx + j*(CARD_W + GAP_X), row_y, CARD_W, CARD_H)
            c.rect = r  # make clickable

            # Use normalizers so keys/values match our colour map
            bonus = ui_card_bonus(c)              # e.g., "diamond"
            points = ui_card_points(c)            # int
            cost = ui_card_cost(c)                # {"diamond": n, ...}
            tier_val = getattr(c, "tier", getattr(c, "rank", "?"))
            tier_colour = WHITE if bonus == "onyx" else BLACK

            # Card shell
            pygame.draw.rect(surface, WHITE, r, border_radius=8)
            pygame.draw.rect(surface, DARKGREY, r, 2, border_radius=8)

            # Header
            header = pygame.Rect(r.x, r.y, r.w, 28)
            pygame.draw.rect(surface, COLOUR_TO_RGB.get(bonus, GREY), header, border_radius=8)

            pts = FONT_BIG.render(str(points) if points else "", True, GOLD)
            surface.blit(pts, (r.x+8, r.y))

            tier = FONT_SMALL.render(f"T{tier_val}", True, tier_colour)
            surface.blit(tier, (r.right-28, r.y+6))

            # Cost pips (use normalized colour keys)
            yy = r.y + 126
            for cc, vv in sorted(cost.items()):
                dot = pygame.Surface((14, 14), pygame.SRCALPHA)
                pygame.draw.circle(dot, COLOUR_TO_RGB.get(cc, GREY), (7, 7), 7)
                surface.blit(dot, (r.x+10, yy+2))
                cost_txt = FONT_SMALL.render(str(vv), True, BLACK)
                surface.blit(cost_txt, (r.x+28, yy-2))
                yy -= 18


def draw_bank_on_board(surface, chip_centres, bank):
    # bank can be TokenBank (with .counts) or plain dict
    counts = bank.counts if hasattr(bank, "counts") else bank
    for colour, centre in chip_centres:
        count = str(counts.get(colour, 0))
        draw_token_chip(surface, centre, colour, count)

# ------------------------ Board Layout Helpers ------------------------

def layout_board(t1: List[Card], t2: List[Card], t3: List[Card], nobles: List[Noble], numPlayers: int):
    # --- Nobles, centred at the top ---
    nobles_show = min(numPlayers+1, len(nobles))
    nobles_row_w = nobles_show*NOBLE_W + (nobles_show-1)*GAP_X
    nobles_x0 = BOARD_LEFT + (BOARD_W - nobles_row_w)//2
    noble_rects = [pygame.Rect(nobles_x0 + i*(NOBLE_W+GAP_X), MARGIN, NOBLE_W, NOBLE_H)
                   for i in range(nobles_show)]
    for nb, r in zip(nobles[:nobles_show], noble_rects): nb.rect = r

    # --- Card rows (Tier 3 top, 2 mid, 1 bottom) ---
    # Each row has: a deck pile at the left + 4 face-up cards; the *whole row group* is centred.
    row_top_y = MARGIN + NOBLE_H + 20
    visible = 4
    rows = []  # each: (deck_rect, card_rects_list)
    for row_idx, deck_source in enumerate([t3, t2, t1]):  # T3, T2, T1
        y = row_top_y + row_idx*(ROW_H + GAP_Y)
        group_w = CARD_W + GAP_X + (visible*CARD_W + (visible-1)*GAP_X)
        left_x = BOARD_LEFT + (BOARD_W - group_w)//2
        deck_rect = pygame.Rect(left_x, y, CARD_W, CARD_H)
        card_rects = [pygame.Rect(left_x + CARD_W + GAP_X + i*(CARD_W+GAP_X), y, CARD_W, CARD_H)
                      for i in range(visible)]
        rows.append((deck_rect, card_rects))
        for c, r in zip(deck_source[:visible], card_rects): c.rect = r

    (deck3_rect, r3), (deck2_rect, r2), (deck1_rect, r1) = rows

    # --- Bottom-centred token bank (single row, correctly clamped) ---
    step = TOKEN_R*2 + 28
    n = len(TOKEN_COLOURS)

    row_visual_w = (n - 1) * step + 2 * TOKEN_R
    left_edge     = BOARD_LEFT + (BOARD_W - row_visual_w) // 2
    first_cx      = left_edge + TOKEN_R

    gap_above_t1  = 28
    lower_bound   = r1[0].bottom + gap_above_t1 + TOKEN_R
    upper_bound   = H - STATUS_H - 8 - TOKEN_R

    if lower_bound <= upper_bound:
        bank_y = upper_bound
    else:
        bank_y = lower_bound

    chip_centres = [(colour, (first_cx + i*step, bank_y)) for i, colour in enumerate(TOKEN_COLOURS)]

    return {
        "nobles": noble_rects,
        "t3": r3, "t2": r2, "t1": r1,
        "deck3": deck3_rect, "deck2": deck2_rect, "deck1": deck1_rect,
        "bank_chips": chip_centres,
    }