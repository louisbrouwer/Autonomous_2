"""Regelgebaseerde agent voor Taxi-v4 (houdt geen rekening met muren).

Logica:
    1. Passagier zit NIET in de taxi:
         - Staat de taxi al op de pickup locatie? -> PICKUP.
         - Anders: beweeg richting de pickup locatie (laagste Manhattan-afstand).
    2. Passagier zit WEL in de taxi:
         - Staat de taxi al op de destination? -> DROPOFF.
         - Anders: beweeg richting de destination.

De agent weet niets van muren, dus loopt er soms tegenaan (geen penalty,
maar wel een stap verspild). Dit is de baseline die een RL-agent moet verslaan.
"""
import random

from ..env import LOCATIONS, decode_state
from .base import Agent

# Bewegingsacties en hun (row, col) verschuiving.
_MOVE_DELTAS = {
    0: (1, 0),    # south
    1: (-1, 0),   # north
    2: (0, 1),    # east
    3: (0, -1),   # west
}
PICKUP, DROPOFF = 4, 5


class RuleBasedAgent(Agent):
    """Agent die op basis van vaste regels actions kiest (geen learning)."""

    n_actions = 6

    def __init__(self, seed: int | None = None) -> None:
        """Maak een rule-based agent aan.

        Args:
            seed: Optioneel getal voor reproduceerbare willekeurige keuzes bij gelijkspel.
        """
        self.rng = random.Random(seed)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Kies een action op basis van de rule-based logica.

        Args:
            state:  Huidige state (gecodeerd als geheel getal).
            greedy: Niet gebruikt; de agent is altijd deterministisch (op ties na).

        Returns:
            Index van de gekozen action.
        """
        row, col, pass_loc, dest = decode_state(state)

        if pass_loc == 4:  # passagier zit al in de taxi -> ga naar destination
            target = LOCATIONS[dest]
            if (row, col) == target:
                return DROPOFF
        else:              # passagier staat nog ergens -> ga hem ophalen
            target = LOCATIONS[pass_loc]
            if (row, col) == target:
                return PICKUP

        return self._greedy_move(row, col, *target)

    def _greedy_move(self, r: int, c: int, tr: int, tc: int) -> int:
        """Kies de beweging die de Manhattan-afstand naar het doel het meest verkleint.

        Bij meerdere even goede actions wordt willekeurig gekozen.

        Args:
            r, c:   Huidige rij en kolom van de taxi.
            tr, tc: Doelrij en doelkolom.

        Returns:
            Index van de beste bewegings-action.
        """
        best_actions: list[int] = []
        best_dist = None
        for action, (dr, dc) in _MOVE_DELTAS.items():
            nr, nc = max(0, min(4, r + dr)), max(0, min(4, c + dc))
            dist = abs(nr - tr) + abs(nc - tc)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_actions = [action]
            elif dist == best_dist:
                best_actions.append(action)
        return self.rng.choice(best_actions)
