"""Hyperparameter-sweep: train een agent over een raster van waarden voor één parameter.

Elke combinatie van parameterwaarde × seed wordt als aparte run uitgevoerd.
Uitvoerstructuur:
    results/<sweep_naam>/<param>=<waarde>_seed=<N>/log.csv
                                                  /qtable.npy

Voorbeeld:
    python -m src.sweep --agent qlearning --param alpha \
        --values 0.1 0.3 0.5 0.9 --seeds 5 --episodes 3000

Combineer daarna met `plot_sweep.py` om gemiddelde ± std curves te maken.
"""
import argparse
import csv
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from .agents.q_learning import QLearningAgent
from .agents.sarsa import SarsaAgent
from .env import N_ACTIONS, N_STATES, TaxiEnv


def _build_agent(name: str, agent_cfg: dict, seed: int):
    """Maak een agent aan op basis van de naam en configuratie.

    Args:
        name:      "qlearning" of "sarsa".
        agent_cfg: Dictionary met agent-parameters (alpha, gamma, enz.).
        seed:      Getal voor reproduceerbaarheid.

    Returns:
        Een geïnitialiseerde agent.
    """
    if name == "qlearning":
        return QLearningAgent(n_states=N_STATES, n_actions=N_ACTIONS,
                              seed=seed, **agent_cfg)
    if name == "sarsa":
        return SarsaAgent(n_states=N_STATES, n_actions=N_ACTIONS,
                          seed=seed, **agent_cfg)
    raise ValueError(f"Onbekende agent: {name}")


def _run_one(agent_name: str, agent_cfg: dict, episodes: int, max_steps: int,
             seed: int, out_dir: Path) -> None:
    """Voer één trainingsrun uit en sla log en Q-table op.

    Args:
        agent_name: "qlearning" of "sarsa".
        agent_cfg:  Agent-parameters voor deze run.
        episodes:   Aantal trainingsepisodes.
        max_steps:  Maximaal aantal stappen per episode.
        seed:       Willekeurig getal voor reproduceerbaarheid.
        out_dir:    Map waar log.csv en qtable.npy naartoe worden geschreven.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = TaxiEnv()
    agent = _build_agent(agent_name, agent_cfg, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "return", "length", "illegal_actions", "delivered", "epsilon"]
        )

        for ep in range(episodes):
            state = env.reset(seed=seed + ep)
            # Kies de eerste action vóór de lus (nodig voor SARSA).
            action = agent.select_action(state)
            ep_return = 0.0
            steps = 0
            illegal = 0
            delivered = 0
            for _ in range(max_steps):
                next_state, reward, done, info = env.step(action)
                ep_return += reward
                steps += 1
                if info.get("illegal_action"):
                    illegal += 1
                if info.get("delivered"):
                    delivered = 1

                if agent_name == "sarsa":
                    # SARSA on-policy: kies de volgende action NU zodat die meegenomen wordt.
                    next_action = agent.select_action(next_state) if not done else None
                    agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action if not done else 0
                else:
                    # Q-learning off-policy: de volgende action hoeft niet mee in de update.
                    agent.update(state, action, reward, next_state, None, done)
                    action = agent.select_action(next_state) if not done else 0

                state = next_state
                if done:
                    break

            agent.end_episode()
            writer.writerow(
                [ep, f"{ep_return:.2f}", steps, illegal, delivered,
                 f"{agent.epsilon:.4f}"]
            )

    agent.save(str(out_dir / "qtable.npy"))
    env.close()


def _coerce(value: str):
    """Zet een CLI-tekstwaarde om naar het juiste type (bool / int / float / str).

    Probeert achtereenvolgens bool, int en float; valt terug op str.
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main() -> None:
    """Hoofdfunctie: lees argumenten, voer de sweep uit en sla het manifest op."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["qlearning", "sarsa"])
    parser.add_argument("--config", type=str,
                        default="experiments/qlearning_default.yaml",
                        help="Basis-configuratie; de sweep overschrijft één veld van 'agent:'.")
    parser.add_argument("--param", required=True,
                        help="Naam van de agent-parameter die gesweept wordt (bijv. alpha, gamma).")
    parser.add_argument("--values", nargs="+", required=True,
                        help="Lijst van waarden voor de gesweepte parameter.")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Aantal seeds per waarde (seeds zijn 0..N-1).")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--sweep-name", type=str, default=None,
                        help="Naam van de sweep-map (standaard: sweep_<agent>_<param>).")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f) or {}
    base_agent_cfg = base_cfg.get("agent", {})

    sweep_name = args.sweep_name or f"sweep_{args.agent}_{args.param}"
    sweep_dir = Path("results") / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    values = [_coerce(v) for v in args.values]
    print(f"Sweeping {args.agent}.{args.param} over {values} met {args.seeds} seeds "
          f"x {args.episodes} episodes -> {sweep_dir}")

    total = len(values) * args.seeds
    done = 0
    start = time.time()
    for value in values:
        for seed in range(args.seeds):
            # Maak een diepe kopie zodat elke run zijn eigen config-dict heeft.
            agent_cfg = deepcopy(base_agent_cfg)
            agent_cfg[args.param] = value
            run_dir = sweep_dir / f"{args.param}={value}_seed={seed}"
            _run_one(args.agent, agent_cfg, args.episodes, args.max_steps,
                     seed, run_dir)
            done += 1
            elapsed = time.time() - start
            eta = elapsed / done * (total - done)
            print(f"  [{done}/{total}] {args.param}={value} seed={seed} "
                  f"({elapsed:.0f}s verstreken, ETA {eta:.0f}s)")

    # Sla een manifest op zodat de sweep later reproduceerbaar is.
    with open(sweep_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(
            {
                "agent": args.agent,
                "config": args.config,
                "param": args.param,
                "values": values,
                "seeds": args.seeds,
                "episodes": args.episodes,
                "base_agent_cfg": base_agent_cfg,
            },
            f,
        )
    print(f"Klaar. Sweep opgeslagen in {sweep_dir}")


if __name__ == "__main__":
    main()
