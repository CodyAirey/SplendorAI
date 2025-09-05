class NobleCard:
    def __init__(self, victoryPoints, emeralds=0, diamonds=0, sapphires=0, onyx=0, rubies=0):
        self.victoryPoints = victoryPoints
        self.emeralds = emeralds
        self.diamonds = diamonds
        self.sapphires = sapphires
        self.onyx = onyx
        self.rubies = rubies

    def __repr__(self):
        return (f"Noble(VictoryPoints={self.victoryPoints}, "
                f"EmeraldReq={self.emeralds}, DiamondReq={self.diamonds}, SapphireReq={self.sapphires}, "
                f"OnyxReq={self.onyx}, RubyReq={self.rubies})")
