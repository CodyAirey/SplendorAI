# ui.py
import pygame
from game_state import GameState
from move_parser import parse_move
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
    Interactive loop with a tiny command-line:
      - Type commands matching your parser, e.g. T(DSE), B(1,2), R(0,3), etc.
      - Enter to submit (parses via parse_move)
      - Backspace to edit; Esc to clear
    Rendering/look stays identical; only the status strip text changes.
    """
    running = True
    cmd_buffer = ""
    last_result = ""   # message after last Enter

    while running:
        CLOCK.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cmd_buffer = ""
                    last_result = ""
                elif event.key == pygame.K_BACKSPACE:
                    cmd_buffer = cmd_buffer[:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    text = cmd_buffer.strip()
                    if text:
                        try:
                            parsed = parse_move(text)
                            last_result = f"OK → {parsed}"
                            # TODO: hand off to controller/executor here, e.g.:
                            # controller.submit(parsed, state)
                            # then re-deal/redraw via draw_once(state, ...)
                        except Exception as e:
                            last_result = f"Error: {e}"
                    cmd_buffer = ""
                else:
                    ch = event.unicode
                    # allow only characters your grammar needs
                    if ch and (ch.isalnum() or ch in "(), -"):
                        cmd_buffer += ch

        # If a custom status provider is given, append its message on the right
        base_status = f"Enter command: {cmd_buffer}"
        dynamic = status_provider() if callable(status_provider) else ""
        status_text = f"{base_status}" + (f"   |   {last_result}" if last_result else "")
        if dynamic:
            status_text += f"   |   {dynamic}"

        draw_once(state, status_text)

    pygame.quit()

