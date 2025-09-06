# ui.py
import pygame
from game_state import GameState
from splendor_template import (
    draw_card, draw_noble, draw_deck_pile,
    draw_bank_on_board, draw_player_panel,
    layout_board, TOKEN_R, MARGIN, H, FONT, WHITE, STATUS_H,
    SCREEN, CLOCK, FPS, W, PANEL_W
)

def init_ui(state: GameState):
    running = True

    # Layout can be computed once since we're only showing the initial state here.
    layout = layout_board(state.deck_t1, state.deck_t2, state.deck_t3, state.nobles, len(state.players))

    while running:
        CLOCK.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        SCREEN.fill((34, 120, 70))

        # Nobles
        for nb in state.nobles[:len(layout["nobles"])]:
            draw_noble(SCREEN, nb)

        # Cards (top 4 of each tier)
        for c in state.deck_t3[:4] + state.deck_t2[:4] + state.deck_t1[:4]:
            draw_card(SCREEN, c)

        # Deck piles + visible cards
        draw_deck_pile(SCREEN, layout["deck3"], 3, len(state.deck_t3))
        draw_deck_pile(SCREEN, layout["deck2"], 2, len(state.deck_t2))
        draw_deck_pile(SCREEN, layout["deck1"], 1, len(state.deck_t1))

        # Bottom bank
        draw_bank_on_board(SCREEN, layout["bank_chips"], state.bank)

        # Status strip (placeholder)
        status = FONT.render("Initialised game", True, WHITE)
        SCREEN.blit(status, (MARGIN, H - STATUS_H + 10))

        # Right panel: players + reserves (correctly positioned)
        panel_x = W - PANEL_W - MARGIN       # <-- use actual panel width
        draw_player_panel(SCREEN, state.players, state.active_idx, panel_x, MARGIN)

        pygame.display.flip()

    pygame.quit()
