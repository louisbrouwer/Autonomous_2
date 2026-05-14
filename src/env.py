"""Dunne wrapper om Gymnasium's Taxi-v3 omgeving.

State space (500 discrete states):
    state = ((taxi_row * 5) + taxi_col) * 5 * 4
            + passenger_location * 4
            + destination
    gedecodeerd als (taxi_row, taxi_col, passenger_loc, destination).

    passenger_loc: 0=R, 1=G, 2=Y, 3=B, 4=in_taxi
    destination  : 0=R, 1=G, 2=Y, 3=B

Action space (6 actions):
    0=south, 1=north, 2=east, 3=west, 4=pickup, 5=dropoff

Rewards:
    -1   per stap
    -10  bij een ongeldige pickup of dropoff
    +20  bij succesvolle aflevering (episode klaar)
"""

import gymnasium as gym

N_STATES = 500
N_ACTIONS = 6
GRID_SHAPE = (5, 5)
ACTION_NAMES = ("south", "north", "east", "west", "pickup", "dropoff")
LOC_NAMES = ("R", "G", "Y", "B", "in_taxi")
LOCATIONS = ((0, 0), (0, 4), (4, 0), (4, 3))   # vaste grid posities voor R, G, Y, B


class TaxiEnv:
    """Wrapper om de Gymnasium Taxi-v3 omgeving met iets eenvoudigere interface."""

    def __init__(self, render_mode: str | None = None) -> None:
        """Maak de omgeving aan.

        Args:
            render_mode: "human" om de omgeving visueel te tonen, None voor geen weergave.
        """
        self.env = gym.make("Taxi-v3", render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed: int | None = None) -> int:
        """Reset de omgeving naar een beginstand en geef de beginstate terug.

        Args:
            seed: Optioneel getal voor een reproduceerbare beginsituatie.

        Returns:
            De beginstate als geheel getal.
        """
        obs, _ = self.env.reset(seed=seed)
        return int(obs)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Voer een action uit en geef het resultaat terug.

        Args:
            action: Index van de uit te voeren action (0–5).

        Returns:
            Tuple van (nieuwe state, reward, done, extra info).
            De info-dict bevat:
                - "illegal_action": True als er een ongeldige pickup/dropoff was.
                - "delivered":      True als de passagier succesvol is afgeleverd.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["illegal_action"] = reward <= -10 and not terminated
        info["delivered"] = reward >= 20
        # Gymnasium splitst "done" op in terminated (doel bereikt) en truncated (timeout).
        done = bool(terminated or truncated)
        return int(obs), float(reward), done, info

    def render(self):
        """Teken de huidige state van de omgeving."""
        return self.env.render()

    def close(self) -> None:
        """Sluit de omgeving netjes af (geef geheugen vrij)."""
        self.env.close()

    def decode(self, state: int) -> tuple[int, int, int, int]:
        """Decodeer een state-getal naar (taxi_row, taxi_col, passenger_loc, destination)."""
        return tuple(self.env.unwrapped.decode(state))  # type: ignore[return-value]


def decode_state(state: int) -> tuple[int, int, int, int]:
    """Decodeer een state-getal zonder dat je een omgevingsobject nodig hebt.

    Args:
        state: State als geheel getal (0–499).

    Returns:
        Tuple (taxi_row, taxi_col, passenger_loc, destination).
    """
    # De state is opgebouwd als geneste gehele-getaldeling; hier pakken we elk deel uit.
    dest = state % 4
    state //= 4
    pass_loc = state % 5
    state //= 5
    col = state % 5
    row = state // 5
    return row, col, pass_loc, dest
