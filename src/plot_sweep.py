"""Maak een samenvattende grafiek van een hyperparameter-sweep.

Leest `results/<sweep_naam>/manifest.yaml` om te weten welke parameter gesweept is
en welke waarden gebruikt zijn. Per waarde worden alle seeds geladen, gemiddeld
en geplot als een curve met een onzekerheidsband (gemiddelde ± standaarddeviatie).
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_returns(run_dir: Path) -> np.ndarray:
    """Laad de episode-rewards uit het log.csv bestand van één run.

    Args:
        run_dir: Map met een "log.csv" bestand.

    Returns:
        NumPy-array met de reward per episode.
    """
    returns: list[float] = []
    with open(run_dir / "log.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            returns.append(float(row["return"]))
    return np.array(returns)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Pas een voortschrijdend gemiddelde toe, ook op 2D-arrays (seeds × episodes).

    Args:
        x:      1D- of 2D-array van rewards.
        window: Window size.

    Returns:
        Afgevlakte array; korter dan de invoer door "valid" convolutie.
    """
    if window <= 1 or x.shape[-1] < window:
        return x
    kernel = np.ones(window) / window
    if x.ndim == 1:
        return np.convolve(x, kernel, mode="valid")
    # 2D: vlak elke rij (= één seed) afzonderlijk af.
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"),
                               axis=-1, arr=x)


def main() -> None:
    """Hoofdfunctie: laad manifest, bereken statistieken en sla de grafiek op."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True, type=str,
                        help="Map met manifest.yaml en de sweep-resultaten.")
    parser.add_argument("--window", type=int, default=50,
                        help="Window size voor het voortschrijdend gemiddelde.")
    parser.add_argument("--output", type=str, default=None,
                        help="Uitvoerpad voor de afbeelding (standaard: sweep_curves.png in sweep-dir).")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    # Lees het manifest om te weten welke parameter en waarden gebruikt zijn.
    with open(sweep_dir / "manifest.yaml") as f:
        manifest = yaml.safe_load(f)
    param = manifest["param"]
    values = manifest["values"]
    seeds = manifest["seeds"]

    plt.figure(figsize=(10, 5))
    for value in values:
        # Laad alle seeds voor deze parameterwaarde.
        runs = []
        for seed in range(seeds):
            run_dir = sweep_dir / f"{param}={value}_seed={seed}"
            runs.append(_load_returns(run_dir))

        # Zorg dat alle runs even lang zijn (knip af op de kortste run).
        n_episodes = min(r.shape[0] for r in runs)
        # Stack tot (seeds, episodes) zodat we per episode kunnen middelen.
        stack = np.stack([r[:n_episodes] for r in runs], axis=0)
        smoothed = _smooth(stack, args.window)
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(mean.shape[0]) + args.window

        line, = plt.plot(x, mean, label=f"{param}={value}")
        # Teken een transparante band voor de spreiding (± 1 standaarddeviatie).
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())

    plt.xlabel("Episode")
    plt.ylabel(f"Reward (window={args.window} eps, gemiddelde ± std over {seeds} seeds)")
    plt.title(f"Sweep over {param} ({manifest['agent']}, Taxi-v3)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = args.output or str(sweep_dir / "sweep_curves.png")
    plt.savefig(output, dpi=120)
    print(f"Grafiek opgeslagen als {output}")


if __name__ == "__main__":
    main()
