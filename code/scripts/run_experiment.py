#!/usr/bin/env python3
"""Run a mechanistic interpretability experiment from config."""

import argparse
import yaml
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['experiment']['seed'])
    print(f"Running experiment {args.experiment}")
    print(f"Config: {config}")

if __name__ == "__main__":
    main()
