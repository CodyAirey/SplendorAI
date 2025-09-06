# main.py
from loader import load_initial_state
from initalize_ui import run_ui

if __name__ == "__main__":
    state = load_initial_state(4)
    run_ui(state)