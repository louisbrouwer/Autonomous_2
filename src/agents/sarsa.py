"""SARSA agent (tabulair, on-policy).

Update-regel:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]

On-policy betekent: we leren op basis van de action die de agent
daadwerkelijk uitvoert (inclusief zijn eigen willekeurige exploration).
Hierdoor is SARSA voorzichtiger dan Q-learning in gevaarlijke situaties.
"""
from .q_learning import QLearningAgent


class SarsaAgent(QLearningAgent):
    """Tabulaire SARSA agent.

    Erft alles van QLearningAgent, maar overschrijft de update-methode
    zodat de volgende action (a') meegenomen wordt in de berekening.
    """

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int | None = None,
               done: bool = False) -> None:
        """Pas de Q-value aan op basis van één experience (on-policy).

        Args:
            state:       Huidige state.
            action:      Uitgevoerde action.
            reward:      Ontvangen reward.
            next_state:  Volgende state.
            next_action: De action die daadwerkelijk in next_state gekozen wordt.
                         Verplicht als de episode nog niet klaar is.
            done:        True als de episode voorbij is.
        """
        if not done and next_action is None:
            raise ValueError("SARSA requires next_action when the episode is not yet done.")
        q_sa = self.Q[state, action]
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] = q_sa + self.alpha * (target - q_sa)
