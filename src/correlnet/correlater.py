"""
Utils to compute corrected correlations.
"""

import pandas as pd
import logging

from scipy.stats import kendalltau, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from typing import Callable, Literal

from correlnet.embedder import Embedder

logger = logging.getLogger(__name__)


class Correlater:
    """Computes corrected pairwise correlations."""

    def __init__(self, method: Literal["pearson", "spearman", "kendall"], correction: str = "bonferroni"):
        self.method = method
        self.correction = correction

    def pairwise_correlations(self, df: pd.DataFrame):
        """Since scipy stats has inconsistent signatures and df.corr does not return pvalues, this method was necessary..."""
        # does not return pvalues unfortunately...
        # cor = df.corr(method=method)
        cor_fn = self.cor_fn()
        cor_dfs = []
        for var_1 in df.columns:
            for var_2 in df.columns:
                logger.debug(f"computing correlation: {var_1},{var_2}")
                statistic, pvalue = cor_fn(df[var_1], df[var_2])
                corr = dict(var_1=var_1, var_2=var_2, statistic=statistic, pvalue=pvalue)
                cor_dfs.append(pd.DataFrame([corr]))
        correl_df = pd.concat(cor_dfs).set_index(["var_1", "var_2"])
        print(correl_df.head())

        if self.correction:
            logger.debug(f"applying {self.correction}")
            _, p_corrected, _, _ = multipletests(correl_df["pvalue"], method=self.correction)
            correl_df["pvalue"] = p_corrected
        print(correl_df.head())
        return correl_df

    def cor_fn(self):
        """Returns the correlation function """
        if self.method == "pearson":
            return pearsonr
        if self.method == "spearman":
            return spearmanr
        if self.method == "kendall":
            return kendalltau
        if callable(self.method):
            return self.method
        raise NotImplementedError(f"method {self.method} is not supported")
