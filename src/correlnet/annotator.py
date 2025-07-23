"""
Annotates Variable nodes with additional information.
"""


import abc

import pandas as pd

from scipy.stats import pearsonr, spearmanr, kendalltau


class Annotator(abc.ABC):
    @abc.abstractmethod
    def annotate(self, df: pd.DataFrame, correl_df: pd.DataFrame):
        pass


class TargetCorrelationAnnotator(Annotator):
    def __init__(self, target, method="pearson"):
        super().__init__()
        self.target = target
        self.method = method

    @abc.abstractmethod
    def annotate(self, df: pd.DataFrame, correl_df: pd.DataFrame) -> pd.Series:
        assert len(self.target) == len(df)
        correl_fn = self.correl_fn()
        # returns dataframe with 2 rows (stat, p) and nvar columns
        target_df = df.apply(lambda x: correl_fn(x, self.target), axis=0)
        return target_df.iloc[0]

    def correl_fn(self):
        if self.method == "pearson":
            return pearsonr
        elif self.method == "spearman":
            return spearmanr
        elif self.method == "kendall":
            return kendalltau
        elif callable(self.method):
            return self.method
        else:
            raise NotImplementedError()
