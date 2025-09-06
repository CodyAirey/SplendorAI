# ui_components.py
# Splendor-style UI scaffold for Pygame with centred nobles, deck piles,
# card grid, bottom-centred coin row, dynamic player panels,
# and a right-side "Reserves" column.

import pygame
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

pygame.init()
pygame.display.set_caption("Splendor Template")

# ------------------------ Config & Constants ------------------------

BASE_W, H = 1280, 780
PANEL_PLAYERS_W = 360
RESERVE_W = 420
W = BASE_W + RESERVE_W

FPS = 60
SCREEN = pygame.display.set_mode((W, H))
CLOCK = pygame.time.Clock()

# Colours
BLACK, WHITE, GREY, DARKGREY, GOLD = (0,0,0), (240,240,240), (200,200,200), (60,60,60), (255,215,0)

TOKEN_COLOURS = ["diamond", "sapphire", "emerald", "ruby", "onyx", "gold"]
COLOUR_TO_RGB = {
    "diamond": (220,220,255), "sapphire": (70,110,255), "emerald": (70,180,120),
    "ruby": (220,60,60), "onyx": (40,40,40), "gold": (240,210,60),
}

# Layout
MARGIN, PANEL_W = 16, PANEL_PLAYERS_W + RESERVE_W
BOARD_LEFT, BOARD_RIGHT = MARGIN, W - PANEL_W - 2*MARGIN
BOARD_W = BOARD_RIGHT - BOARD_LEFT
ROW_H, CARD_W, CARD_H = 160, 110, 150
NOBLE_W, NOBLE_H = 110, 110
TOKEN_R, GAP_X, GAP_Y, STATUS_H = 36, 20, 12, 44

# Fonts
FONT = pygame.font.SysFont("arial", 18)
FONT_SMALL = pygame.font.SysFont("arial", 14)
FONT_SMALL_BOLD = pygame.font.SysFont("arial", 14, bold=True)
FONT_BIG = pygame.font.SysFont("arial", 22, bold=True)


# ------------------------ Data Models (UI-only shims) ------------------------

@dataclass
class Card:
    tier: int
    points: int
    bonus: str
    cost: Dict[str, int]
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0,0,CARD_W,CARD_H))

@dataclass
class Noble:
    points: int
    requirement: Dict[str, int]
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0,0,NOBLE_W,NOBLE_H))

@dataclass
class Player:
    name: str
    tokens: Dict[str, int]
    bonuses: Dict[str, int]
    points: int = 0
    reserved: List[Card] = field(default_factory=list)

@dataclass
class TokenBank:
    counts: Dict[str, int]


# ------------------------ Small Drawing Helpers ------------------------

def _draw_circle_with_text(surface, colour, pos, radius, text, text_colour, font=FONT_SMALL_BOLD):
    pygame.draw.circle(surface, colour, pos, radius)
    pygame.draw.circle(surface, DARKGREY, pos, radius, 2)
    txt = font.render(str(text), True, text_colour)
    surface.blit(txt, txt.get_rect(center=pos))


def _draw_square_with_text(surface, colour, rect, text, text_colour):
    pygame.draw.rect(surface, colour, rect, border_radius=4)
    pygame.draw.rect(surface, DARKGREY, rect, 2, border_radius=4)
    txt = FONT_SMALL_BOLD.render(str(text), True, text_colour)
    surface.blit(txt, txt.get_rect(center=rect.center))


def _draw_cost_pip(surface, x, y, colour, value):
    dot = pygame.Surface((16, 16), pygame.SRCALPHA)
    pygame.draw.circle(dot, COLOUR_TO_RGB.get(colour, GREY), (8,8), 8)
    surface.blit(dot, (x, y))
    txt = FONT_SMALL.render(str(value), True, BLACK)
    surface.blit(txt, (x+20, y))


# ------------------------ Rendering ------------------------

def draw_token_chip(surface, centre: Tuple[int,int], colour: str, label: str):
    _draw_circle_with_text(
        surface,
        COLOUR_TO_RGB[colour],
        centre,
        TOKEN_R,
        label,
        BLACK if colour != "onyx" else WHITE,
        font=FONT_BIG   # bigger font for bank chips
    )


def draw_card(surface, card):
    pygame.draw.rect(surface, WHITE, card.rect, border_radius=8)
    pygame.draw.rect(surface, DARKGREY, card.rect, 2, border_radius=8)

    bonus = card.gemType.lower()
    points = card.victoryPoints
    cost = {k.lower(): v for k,v in card.cost.items() if v > 0}

    # Header
    header = pygame.Rect(card.rect.x, card.rect.y, card.rect.w, 28)
    pygame.draw.rect(surface, COLOUR_TO_RGB.get(bonus, GREY), header, border_radius=8)

    if points:
        surface.blit(FONT_BIG.render(str(points), True, GOLD), (card.rect.x+8, card.rect.y))

    tier_val = card.rank or "?"
    tier_colour = WHITE if bonus == "onyx" else BLACK
    surface.blit(FONT_SMALL.render(f"T{tier_val}", True, tier_colour), (card.rect.right-28, card.rect.y+6))

    y = card.rect.y + 126
    for c, v in sorted(cost.items()):
        _draw_cost_pip(surface, card.rect.x+10, y+2, c, v)
        y -= 20


def draw_noble(surface, noble):
    pygame.draw.rect(surface, (245,235,210), noble.rect, border_radius=8)
    pygame.draw.rect(surface, DARKGREY, noble.rect, 2, border_radius=8)

    pts_val = noble.victoryPoints
    reqs = {
        "emerald": noble.emerald,
        "diamond": noble.diamond,
        "sapphire": noble.sapphire,
        "onyx": noble.onyx,
        "ruby": noble.ruby,
    }
    reqs = {k:v for k,v in reqs.items() if v > 0}

    surface.blit(FONT_SMALL.render("Noble", True, BLACK), (noble.rect.x+68, noble.rect.y+8))
    surface.blit(FONT_BIG.render(str(pts_val), True, BLACK), (noble.rect.x+8, noble.rect.y+8))

    y = noble.rect.y + 90
    for c,v in sorted(reqs.items()):
        _draw_cost_pip(surface, noble.rect.x+10, y, c, v)
        y -= 18


def draw_deck_pile(surface, rect: pygame.Rect, tier: int, remaining: int):
    pygame.draw.rect(surface, (210,200,170), rect, border_radius=8)
    pygame.draw.rect(surface, DARKGREY, rect, 2, border_radius=8)
    surface.blit(FONT_SMALL.render(f"T{tier}", True, BLACK), (rect.x+8, rect.y+6))
    txt = FONT_BIG.render(str(remaining), True, BLACK)
    surface.blit(txt, txt.get_rect(center=(rect.centerx, rect.centery+8)))


def draw_player_panel(surface, players: List[Player], active_idx: int, x0: int, y0: int):
    pygame.draw.rect(surface, (250,250,255), pygame.Rect(x0,0,PANEL_W,H))
    surface.blit(FONT_BIG.render("Players", True, BLACK), (x0+12, y0))
    surface.blit(FONT_BIG.render("Reserves", True, BLACK), (x0+PANEL_PLAYERS_W+12, y0))

    block_h, pad = 150, 16
    for idx, p in enumerate(players[:4]):
        block = pygame.Rect(x0+12, y0+30+idx*(block_h+pad), PANEL_PLAYERS_W-24, block_h)
        pygame.draw.rect(surface, WHITE, block, border_radius=8)
        pygame.draw.rect(surface, DARKGREY, block, 2, border_radius=8)

        prefix = "> " if idx==active_idx else ""
        surface.blit(FONT.render(f"{prefix}{p.name}  (Pts {p.points})", True, BLACK), (block.x+12, block.y+8))

        step = max(40, (block.w-24)//len(TOKEN_COLOURS))
        start_x = block.x+12+step//2
        cy_tokens, cy_bonus = block.y+52, block.y+84

        for i,c in enumerate(TOKEN_COLOURS):
            cx = start_x+i*step
            _draw_circle_with_text(surface, COLOUR_TO_RGB[c], (cx,cy_tokens), 13,
                                   p.tokens.get(c,0), WHITE if c=="onyx" else BLACK)
        for i,c in enumerate([k for k in TOKEN_COLOURS if k!="gold"]):
            cx = start_x+i*step
            rect = pygame.Rect(cx-10, cy_bonus-10, 20, 20)
            _draw_square_with_text(surface, COLOUR_TO_RGB[c], rect,
                                   p.bonuses.get(c,0), WHITE if c=="onyx" else BLACK)

        total = sum(p.tokens.values())
        surface.blit(FONT_SMALL.render(f"Total coins: {total}", True, BLACK),
                     (block.centerx-40, cy_bonus+34))

        row_y, start_rx = block.y+(block_h-CARD_H)//2+2, x0+PANEL_PLAYERS_W+12
        for j,c in enumerate(p.reserved[:3]):
            c.rect = pygame.Rect(start_rx+j*(CARD_W+GAP_X), row_y, CARD_W, CARD_H)
            draw_card(surface, c)


def draw_bank_on_board(surface, chip_centres, bank):
    counts = bank.counts if hasattr(bank, "counts") else bank
    for colour, centre in chip_centres:
        draw_token_chip(surface, centre, colour, str(counts.get(colour,0)))


# ------------------------ Layout ------------------------

def layout_board(t1: List[Card], t2: List[Card], t3: List[Card],
                 nobles: List[Noble], numPlayers: int) -> Dict[str, object]:
    nobles_show = min(numPlayers+1, len(nobles))
    row_w = nobles_show*NOBLE_W+(nobles_show-1)*GAP_X
    x0 = BOARD_LEFT+(BOARD_W-row_w)//2
    noble_rects = [pygame.Rect(x0+i*(NOBLE_W+GAP_X), MARGIN, NOBLE_W, NOBLE_H) for i in range(nobles_show)]
    for nb,r in zip(nobles[:nobles_show], noble_rects): nb.rect=r

    row_top_y, max_visible = MARGIN+NOBLE_H+20, 4
    rows=[]
    for row_idx, tableau in enumerate([t3,t2,t1]):
        y = row_top_y+row_idx*(ROW_H+GAP_Y)
        group_w = CARD_W+GAP_X+(max_visible*CARD_W+(max_visible-1)*GAP_X)
        left_x = BOARD_LEFT+(BOARD_W-group_w)//2
        deck_rect=pygame.Rect(left_x,y,CARD_W,CARD_H)

        visible=min(max_visible,len(tableau))
        rects=[pygame.Rect(left_x+CARD_W+GAP_X+i*(CARD_W+GAP_X),y,CARD_W,CARD_H) for i in range(visible)]
        for c,r in zip(tableau[:visible],rects): c.rect=r
        rows.append((deck_rect,rects))

    (deck3_rect,r3),(deck2_rect,r2),(deck1_rect,r1)=rows

    step=TOKEN_R*2+28
    row_visual_w=(len(TOKEN_COLOURS)-1)*step+2*TOKEN_R
    left_edge=BOARD_LEFT+(BOARD_W-row_visual_w)//2
    first_cx=left_edge+TOKEN_R

    tier1_bottom=row_top_y+2*(ROW_H+GAP_Y)+CARD_H
    bank_y=min(H-STATUS_H-8-TOKEN_R, tier1_bottom+28+TOKEN_R)
    chip_centres=[(c,(first_cx+i*step,bank_y)) for i,c in enumerate(TOKEN_COLOURS)]

    return {"nobles":noble_rects,"t3":r3,"t2":r2,"t1":r1,
            "deck3":deck3_rect,"deck2":deck2_rect,"deck1":deck1_rect,
            "bank_chips":chip_centres}
