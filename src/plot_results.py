"""Maak een grafiek van de reward curves van één of meer trainingsruns."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_returns(run_dir: Path) -> np.ndarray:
    """Laad de episode-rewards uit het log.csv bestand van een trainingsrun.

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


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Pas een voortschrijdend gemiddelde toe om de curve vlakker te maken.

    Args:
        x:      Tijdreeks van rewards.
        window: Window size; bij 1 of kleiner wordt niets gedaan.

    Returns:
        Afgevlakte reeks (korter dan de invoer door "valid" convolutie).
    """
    if window <= 1 or len(x) < window:
        return x
    # np.convolve met "valid" geeft alleen overlap waar het venster volledig past.
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main() -> None:
    """Hoofdfunctie: laad runs, teken curves en sla de grafiek op."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Een of meer mappen met log.csv bestanden.")
    parser.add_argument("--window", type=int, default=20,
                        help="Window size voor het voortschrijdend gemiddelde.")
    parser.add_argument("--output", type=str, default="results/reward_curves.png")
    args = parser.parse_args()

    plt.figure(figsize=(9, 5))
    for run in args.runs:
        run_dir = Path(run)
        returns = load_returns(run_dir)
        smoothed = smooth(returns, args.window)
        # Verschuif de x-as zodat episode 0 overeenkomt met het einde van het eerste venster.
        x = np.arange(len(smoothed)) + args.window
        plt.plot(x, smoothed, label=run_dir.name)

    plt.xlabel("Episode")
    plt.ylabel(f"Reward (voortschrijdend gemiddelde, window={args.window})")
    plt.title("Trainings-reward curves — Taxi-v3")
    # Stippellijn als referentie voor de benaderde optimale score (~8 punten).
    plt.axhline(8, ls="--", c="grey", alpha=0.5,
                label="~optimaal (~8: 12 stappen + 20 aflevering)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"Grafiek opgeslagen als {args.output}")


if __name__ == "__main__":
    main()
