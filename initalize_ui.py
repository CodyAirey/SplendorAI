# ui.py
import pygame
from game_state import GameState
from move_parser import parse_move
from engine import apply_move
from ui_components import (
    draw_card, draw_noble, draw_deck_pile,
    draw_bank_on_board, draw_player_panel,
    layout_board, TOKEN_R, MARGIN, H, FONT, WHITE, STATUS_H,
    SCREEN, CLOCK, FPS, W, PANEL_W
)

def init_ui(state: GameState):
    running = True

    # Layout can be computed once since we're only showing the initial state here.
    layout = layout_board(state.table_t1, state.table_t2, state.table_t3, state.nobles, len(state.players))

    while running:
        CLOCK.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        SCREEN.fill((34, 120, 70))

        # Nobles
        for nb in state.nobles[:len(layout["nobles"])]:
            draw_noble(SCREEN, nb)

        # Cards on table
        for c in state.table_t1 + state.table_t2 + state.table_t3:
            draw_card(SCREEN, c)

        # Deck piles + visible cards
        if state.deck_t3: draw_deck_pile(SCREEN, layout["deck3"], 3, len(state.deck_t3))
        if state.deck_t2: draw_deck_pile(SCREEN, layout["deck2"], 2, len(state.deck_t2))
        if state.deck_t1: draw_deck_pile(SCREEN, layout["deck1"], 1, len(state.deck_t1))

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



def draw_once(state: GameState, status_text: str = "Initialised game"):
    """
    Render a single frame based on the current state.
    Recomputes layout every call so tableau/deck changes are reflected.
    """
    # Layout depends on current tableau lengths & player count
    layout = layout_board(state.table_t1, state.table_t2, state.table_t3,
                          state.nobles, len(state.players))

    SCREEN.fill((34, 120, 70))

    # Nobles
    for nb in state.nobles[:len(layout["nobles"])]:
        draw_noble(SCREEN, nb)

    # Face-up cards on table (whatever is currently there)
    for c in state.table_t3 + state.table_t2 + state.table_t1:
        draw_card(SCREEN, c)

    # Deck piles (only draw if deck still has cards)
    if state.deck_t3: draw_deck_pile(SCREEN, layout["deck3"], 3, len(state.deck_t3))
    if state.deck_t2: draw_deck_pile(SCREEN, layout["deck2"], 2, len(state.deck_t2))
    if state.deck_t1: draw_deck_pile(SCREEN, layout["deck1"], 1, len(state.deck_t1))

    # Bottom bank
    draw_bank_on_board(SCREEN, layout["bank_chips"], state.bank)

    # Status strip
    status = FONT.render(status_text, True, WHITE)
    SCREEN.blit(status, (MARGIN, H - STATUS_H + 10))

    # Right panel: players + reserves
    panel_x = W - PANEL_W - MARGIN
    draw_player_panel(SCREEN, state.players, state.active_idx, panel_x, MARGIN)

    pygame.display.flip()


def run_ui(state: GameState, status_provider=None):
    """
    Interactive loop:
      - Type commands per your parser (e.g., T(DSE), T(DD), R(0,3), B(2,1))
      - Enter to submit; engine mutates state; UI re-renders.
    """
    running = True
    cmd_buffer = ""
    last_result = "Enter command"

    while running:
        CLOCK.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if state.game_over:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                continue
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cmd_buffer = ""
                    last_result = "Enter command"
                elif event.key == pygame.K_BACKSPACE:
                    cmd_buffer = cmd_buffer[:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    text = cmd_buffer.strip()
                    if text:
                        try:
                            parsed = parse_move(text)
                            # apply to state
                            last_result = apply_move(state, parsed)
                        except Exception as e:
                            last_result = f"Error: {e}"
                    cmd_buffer = ""
                else:
                    ch = event.unicode
                    if ch and (ch.isalnum() or ch in "(), -"):
                        cmd_buffer += ch

        dynamic = status_provider() if callable(status_provider) else ""
        if state.game_over and state.final_summary:
            last_result = state.final_summary  # pin the winner message
        status_text = f"Enter command: {cmd_buffer}" + (f"   |   {last_result}" if last_result else "")
        if dynamic:
            status_text += f"   |   {dynamic}"
        draw_once(state, status_text)

    pygame.quit()
