class Noble:
    def __init__(self, victoryPoints, emeralds=0, diamonds=0, sapphires=0, onyx=0, rubies=0):
        self.victoryPoints = victoryPoints
        self.emerald = emeralds
        self.diamond = diamonds
        self.sapphire = sapphires
        self.onyx = onyx
        self.ruby = rubies

    def __repr__(self):
        return (f"Noble(VictoryPoints={self.victoryPoints}, "
                f"EmeraldReq={self.emerald}, DiamondReq={self.diamond}, SapphireReq={self.sapphire}, "
                f"OnyxReq={self.onyx}, RubyReq={self.ruby})")
