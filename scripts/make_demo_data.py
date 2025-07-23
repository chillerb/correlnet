#!/usr/bin/env python

"""
Generates Random Data to demonstrate correlnet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=200, help="number of observations")
    parser.add_argument("-p", type=int, default=50, help="number of features")
    parser.add_argument("-o", "--output", type=Path, default=Path("demo_data.csv"), help="output path")
    parser.add_argument("--seed", type=int, default=19, help="random seed")

    args = parser.parse_args()
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(args)
    rng = np.random.default_rng(args.seed)

    mean = np.zeros(args.p)
    cov = rng.normal(size=(args.p, args.p))
    # ensure symmetry and positive semi-definiteness
    cov = cov @ cov.T
    x = rng.multivariate_normal(mean=mean, cov=cov, size=args.n)
    logger.info(f"generating normal data ({x.shape})")

    df = pd.DataFrame(x, columns=[f"x_{i + 1:0>2}" for i in range(args.p)])

    logger.info(f"writing csv to {args.output}")
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    logging.basicConfig()
    main()
