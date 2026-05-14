"""Visualiseer een geleerde Taxi-poli als pijlenrasters en waarde-heatmaps.

De toestandsruimte heeft 500 toestanden (25 taxiposities × 5 passagierslocaties × 4 bestemmingen).
Alles tegelijk tonen is te druk, dus we maken een 2×4 paneel:
  - 2 rijen: passagier wacht (rij 0) vs. passagier al in taxi (rij 1).
  - 4 kolommen: de vier mogelijke bestemmingen (R, G, Y, B).

Elke cel toont een 5×5 grid met:
  - Kleurkaart (heatmap) van de waarde V(s) = max_a Q(s, a).
  - Pijlen voor bewegings-actions en letters P/D voor pickup/dropoff.
  - Oranje cirkel = passagierslocatie, rood vierkant = bestemming.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from .env import GRID_SHAPE, LOC_NAMES, LOCATIONS

# Verschuiving in (rij, kolom) per bewegingsactie; pickup/dropoff krijgen een letter.
_ARROW = {
    0: (1, 0),    # naar beneden (south)
    1: (-1, 0),   # naar boven (north)
    2: (0, 1),    # naar rechts (east)
    3: (0, -1),   # naar links (west)
}
_PICKUP, _DROPOFF = 4, 5


def _encode(row: int, col: int, pass_loc: int, dest: int) -> int:
    """Codeer (row, col, passagier, destination) terug naar één state-getal."""
    return ((row * 5 + col) * 5 + pass_loc) * 4 + dest


def _draw_cell(ax, Q: np.ndarray, pass_loc: int, dest: int, title: str) -> None:
    """Teken één cel van het paneel: heatmap + actions voor een vaste passagier/destination-combinatie.

    Args:
        ax:       Matplotlib-as om op te tekenen.
        Q:        Q-table met vorm (500, 6).
        pass_loc: Passagierslocatie (0–3 = wachtend, 4 = in taxi).
        dest:     Destination-index (0–3).
        title:    Titel boven de cel.
    """
    n_rows, n_cols = GRID_SHAPE
    V = np.zeros((n_rows, n_cols))
    greedy = np.zeros((n_rows, n_cols), dtype=int)

    # Bereken voor elke taxipositie de waarde V(s) en de greedy action.
    for r in range(n_rows):
        for c in range(n_cols):
            s = _encode(r, c, pass_loc, dest)
            V[r, c] = Q[s].max()
            greedy[r, c] = int(Q[s].argmax())

    # Teken de heatmap (hogere waarde = lichter/geler).
    im = ax.imshow(V, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    # Teken pijlen of letters bovenop de heatmap.
    for r in range(n_rows):
        for c in range(n_cols):
            a = greedy[r, c]
            if a in _ARROW:
                dy, dx = _ARROW[a]
                ax.arrow(c, r, 0.3 * dx, 0.3 * dy,
                         head_width=0.15, head_length=0.15,
                         fc="white", ec="black", length_includes_head=True)
            elif a == _PICKUP:
                ax.text(c, r, "P", color="white", ha="center", va="center",
                        fontsize=10, fontweight="bold")
            elif a == _DROPOFF:
                ax.text(c, r, "D", color="white", ha="center", va="center",
                        fontsize=10, fontweight="bold")

    # Teken een oranje cirkel op de passagierslocatie (alleen als die buiten de taxi is).
    if pass_loc < 4:
        pr, pc = LOCATIONS[pass_loc]
        ax.add_patch(plt.Circle((pc, pr), 0.35, fill=False,
                                edgecolor="orange", linewidth=2))

    # Teken een rood vierkant op de bestemming.
    dr, dc = LOCATIONS[dest]
    ax.add_patch(plt.Rectangle((dc - 0.4, dr - 0.4), 0.8, 0.8, fill=False,
                               edgecolor="red", linewidth=2))

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_title(title, fontsize=10)


def main() -> None:
    """Hoofdfunctie: laad Q-tabel, maak het 2×4 paneel en sla de afbeelding op."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Pad naar de opgeslagen Q-table (.npy).")
    parser.add_argument("--output", required=True,
                        help="Pad waar de afbeelding opgeslagen wordt.")
    parser.add_argument("--title", default="Geleerd Taxi-beleid")
    args = parser.parse_args()

    Q = np.load(args.weights)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for dest in range(4):
        # Rij 0: passagier wacht bij R (pass_loc=0). Rij 1: passagier al in de taxi (pass_loc=4).
        _draw_cell(axes[0, dest], Q, pass_loc=0, dest=dest,
                   title=f"Ophalen bij R, afzetten bij {LOC_NAMES[dest]}")
        _draw_cell(axes[1, dest], Q, pass_loc=4, dest=dest,
                   title=f"In taxi, afzetten bij {LOC_NAMES[dest]}")

    fig.suptitle(args.title + "  (oranje=passagier, rood=bestemming)", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"Beleidsplot opgeslagen als {args.output}")


if __name__ == "__main__":
    main()
