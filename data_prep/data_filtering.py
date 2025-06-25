class GermanPresenceFilter:
    def __init__(self, strategy="ratio", min_ratio=0.5, min_count=3):
        self.strategy = strategy
        self.min_ratio = min_ratio
        self.min_count = min_count

        self.en_tags = {"1", "3a", "3a-E", "4", "4-E"}
        self.de_tags = {"2", "3a-D", "4-D"}

    def _count_de_tokens(self, lang_ids):
        return sum(1 for tag in lang_ids if tag in self.de_tags)

    def apply(self, lang_ids):
        total = len(lang_ids)
        if total == 0:
            return False

        de_count = self._count_de_tokens(lang_ids)
        ratio = de_count / total

        if self.strategy == "ratio":
            return ratio >= self.min_ratio
        elif self.strategy == "threshold":
            return de_count >= self.min_count
        elif self.strategy == "hybrid":
            return de_count >= self.min_count and ratio >= self.min_ratio
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
