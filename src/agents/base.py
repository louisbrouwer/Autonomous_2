"""Basis-interface voor alle agents."""

class Agent:
    """Basisklasse waar alle agents van erven.

    Elke agent moet op zijn minst `select_action` implementeren.
    De andere methoden hebben een lege standaardimplementatie zodat
    subklassen alleen overschrijven wat ze nodig hebben.
    """

    n_actions: int = 6

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Kies een action voor de gegeven state.

        Args:
            state:  De huidige state (geheel getal).
            greedy: Als True, kies altijd de beste bekende action (geen exploration).

        Returns:
            Index van de gekozen action.
        """
        raise NotImplementedError

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int | None, done: bool) -> None:
        """Verwerk één stap en pas de interne kennis aan (optioneel)."""
        return None

    def end_episode(self) -> None:
        """Wordt aangeroepen aan het einde van een episode (optioneel)."""
        return None

    def save(self, path: str) -> None:
        """Sla de agent op naar een bestand (optioneel)."""
        return None

    def load(self, path: str) -> None:
        """Laad de agent vanuit een bestand (optioneel)."""
        return None
