#!/usr/bin/env python

"""
Script to draw a correlation network for a CSV file.
"""

from correlnet.correlnet import CorrelNet, correlnet
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def main():
    """CLI that draws a correlation network for a given csv file."""
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=Path, help="input csv")
    parser.add_argument("--columns", type=str, nargs="+", help="columns to plot")
    parser.add_argument("--index-col", action="store_true", help="index column?")
    parser.add_argument("--title", type=str, default="Correlation Network", help="figure title")
    parser.add_argument("--pos", choices=["random", "var_tsne", "correl_tsne"], default="var_tsne", help="method to compute node positions")
    parser.add_argument("--method", choices=["pearson", "spearman", "kendall"], default="pearson", help="method to compute correlations")
    parser.add_argument("--alpha", type=float, default=0.05, help="significance threshold")
    parser.add_argument("--perplexity", type=int, default=30, help="tsne perplexity")
    parser.add_argument("--seed", type=int, default=19, help="rng seed")
    parser.add_argument("--verbose", action="store_true", help="show debug info")
    parser.add_argument("-o", "--output", type=Path, default=Path("correlnet.png"), help="index column?")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.debug(f"creating output path {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, index_col=args.index_col)

    tsne_kwargs = {"perplexity": args.perplexity}
    cg = correlnet(df, tsne_kwargs=tsne_kwargs, random_state=args.seed, alpha=args.alpha, method=args.method, embedding=args.pos)

    print("plotting")

    cg.plot()

    plt.title(args.title)

    logger.info(f"writing figure to {args.output}")
    plt.savefig(args.output)
    plt.close()


if __name__ == "__main__":
    main()
