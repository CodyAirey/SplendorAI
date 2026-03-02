class CardCost:
    def __init__(self, ruby=0, onyx=0, sapphire=0, diamond=0, emerald=0):
        self._costs = {
            "Diamond": diamond,
            "Sapphire": sapphire,
            "Emerald": emerald,
            "Ruby": ruby,
            "Onyx": onyx,
        }

    def __getitem__(self, gem):
        return self._costs[gem]

    def __setitem__(self, gem, value):
        self._costs[gem] = value

    def __iter__(self):
        return iter(self._costs.values())

    def items(self):
        return self._costs.items()

    def __repr__(self):
        parts = [f"{gem}: {amt}" for gem, amt in self._costs.items() if amt > 0]
        return "{" + ", ".join(parts) + "}" if parts else "{Free}"



class Card:
    def __init__(self, gemType, victoryPoints, cost: CardCost, card_id=None, rank=None):
        self.id = card_id
        self.rank = rank
        self.gemType = gemType
        self.victoryPoints = victoryPoints
        self.cost = cost

    def __repr__(self):
        id_part = f"{self.id}, " if self.id else ""
        rank_part = f"Rank={self.rank}, " if self.rank is not None else ""
        return (f"Card({id_part}{rank_part}gemType={self.gemType}, "
                f"victoryPoints={self.victoryPoints}, cost={self.cost}, "
                f"valProp={round(self.getCostPerPoint(), 2)})")

    def getCostPerPoint(self):
        if self.victoryPoints == 0:
            return float("inf")
        return sum(self.cost) / self.victoryPoints

    def getValuePerCost(self):
        total_cost = sum(self.cost)
        return self.victoryPoints / total_cost if total_cost else float("inf")