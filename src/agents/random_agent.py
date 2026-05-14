"""Willekeurige agent als simpele baseline."""

import random

from .base import Agent


class RandomAgent(Agent):
    """Kiest bij elke stap een willekeurige action. Leert niets."""

    def __init__(self, n_actions: int = 6, seed: int | None = None) -> None:
        """Maak een willekeurige agent aan.

        Args:
            n_actions: Aantal mogelijke actions.
            seed:      Optioneel voor reproduceerbare resultaten.
        """
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Kies een willekeurige action (state en greedy-vlag worden genegeerd).

        Returns:
            Een willekeurige action-index.
        """
        return self.rng.randrange(self.n_actions)
