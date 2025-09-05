class GameBoard:
    def __init__(self, num_players: int):
        # Setup gems based on player count
        # Splendor rules: 2p=4 gems each, 3p=5 gems, 4p=7 gems
        if num_players == 2:
            base = 4
        elif num_players == 3:
            base = 5
        else:
            base = 7

        self.Ruby = base
        self.Onyx = base
        self.Sapphire = base
        self.Diamond = base
        self.Emerald = base
        self.Gold = 5  # always 5 gold jokers

        # nobles/cards will be filled later
        self.nobles = []
        self.cards = {1: [], 2: [], 3: []}