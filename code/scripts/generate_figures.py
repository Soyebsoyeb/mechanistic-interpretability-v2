#!/usr/bin/env python3
"""Generate all figures for the book from raw results."""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def generate_all_figures(results_dir="experiments", output_dir="book/figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for result_file in Path(results_dir).glob("**/*.json"):
        with open(result_file) as f:
            data = json.load(f)
        fig = generate_figure(data)
        fig.savefig(output_dir / f"{result_file.stem}.png", dpi=300)
        plt.close(fig)

def generate_figure(data):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f"Experiment: {data.get('experiment_id', 'unknown')}",
            ha='center', va='center')
    return fig

if __name__ == "__main__":
    generate_all_figures()
