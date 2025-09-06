# main.py
from loader import load_initial_state
from initalize_ui import init_ui

if __name__ == "__main__":
    # state = load_initial_state()
    # run_ui(state)

    state = load_initial_state(4)
    init_ui(state)
