# Taxi-v3 RL (Q-learning en SARSA)

RL-project op Gymnasium Taxi-v3. We vergelijken:

- Q-learning (tabulair)
- SARSA (tabulair)
- baselines: random en rule-based

## Installatie

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Structuur

```text
.
├── src/
│   ├── env.py                # Taxi-v3 wrapper en state-decoder
│   ├── agents/
│   │   ├── base.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── q_learning.py     # tabulaire Q-learning
│   │   └── sarsa.py          # tabulaire SARSA
│   ├── train.py              # training
│   ├── evaluate.py           # evaluatie
│   ├── plot_results.py       # training curves
│   └── plot_policy.py        # policy visualisatie
├── experiments/              # YAML configs voor hyperparameters
├── results/                  # logs, Q-tables, plots
├── tests/                    # kleine tests
└── report/                   # verslag
```

## Korte samenvatting

- States: 500 discrete staten (taxi positie, passagier locatie, bestemming).
- Acties (6): 0=zuid, 1=noord, 2=oost, 3=west, 4=pickup, 5=dropoff.
- Rewards: -1 per stap, -10 bij illegale pickup/dropoff, +20 bij succesvolle episode.
