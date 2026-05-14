"""Q-learning agent (tabulair, off-policy).

Update-regel:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Off-policy betekent: we leren de beste action in de volgende state,
ongeacht welke action de epsilon-greedy strategie daadwerkelijk kiest.
"""

import random

import numpy as np

from .base import Agent


class QLearningAgent(Agent):
    """Tabulaire Q-learning agent.

    Houdt voor elke (state, action)-combinatie een Q-value bij in een Q-table.
    Tijdens training wordt epsilon-greedy gebruikt voor exploration.
    """

    def __init__(
        self,
        n_states: int = 500,
        n_actions: int = 6,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 2000,
        seed: int | None = None,
    ) -> None:
        """Maak een nieuwe Q-learning agent aan.

        Args:
            n_states:               Aantal states in de omgeving.
            n_actions:              Aantal mogelijke actions.
            alpha:                  Learning rate (0 < alpha <= 1).
            gamma:                  Discount factor voor toekomstige rewards.
            epsilon_start:          Beginwaarde van epsilon (veel exploration).
            epsilon_end:            Eindwaarde van epsilon (weinig exploration).
            epsilon_decay_episodes: Na hoeveel episodes epsilon zijn eindwaarde bereikt.
            seed:                   Optioneel voor reproduceerbare resultaten.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self._episode = 0
        self.rng = random.Random(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Kies een action via epsilon-greedy strategie.

        Args:
            state:  Huidige state.
            greedy: Als True, altijd de beste action kiezen (geen random).

        Returns:
            Index van de gekozen action.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self._argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int | None = None,
               done: bool = False) -> None:
        """Pas de Q-value aan op basis van één experience.

        Args:
            state:       Huidige state.
            action:      Uitgevoerde action.
            reward:      Ontvangen reward.
            next_state:  Volgende state.
            next_action: Niet gebruikt bij Q-learning (off-policy).
            done:        True als de episode voorbij is.
        """
        q_sa = self.Q[state, action]
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state].max()
        self.Q[state, action] = q_sa + self.alpha * (target - q_sa)

    def end_episode(self) -> None:
        """Verlaag epsilon lineair na elke episode (minder exploration)."""
        self._episode += 1
        frac = min(1.0, self._episode / max(1, self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _argmax(self, values: np.ndarray) -> int:
        """Geef de index terug van de hoogste value; bij gelijkspel random kiezen."""
        max_v = values.max()
        candidates = np.flatnonzero(values == max_v)
        return int(self.rng.choice(candidates.tolist()))

    def save(self, path: str) -> None:
        """Sla de Q-table op als .npy bestand."""
        np.save(path, self.Q)

    def load(self, path: str) -> None:
        """Laad een eerder opgeslagen Q-table."""
        self.Q = np.load(path)
